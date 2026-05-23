import html
import json
import logging
import logging as stdlib_logging
from copy import deepcopy
from typing import (
    Any,
)

import argilla as rg
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.models.base_clients.openai import SecretStr
from distilabel.models.llms.utils import prepare_output
from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path
from distilabel.typing import FormattedInput, GenerateOutput
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from typing_extensions import override

from linalg_zero.config.data import (
    DistillationConfig,
    LlamaCppServerConfig,
    VllmServerConfig,
)
from linalg_zero.distillation.components.models import (
    ModelParameters,
    ModelType,
)
from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.system_prompts import THINK_CLOSE, THINK_OPEN
from linalg_zero.shared.utils import get_libpath, get_logger, setup_logging

logger = get_logger(__name__)


# TODO: is this the right file to store this class in?
class CustomOpenAILLM(OpenAILLM):
    """
    Patched OpenAI LLM that supports tool calls by bypassing the restrictive validation.
    This allows using the full OpenAI API format with tool_calls and tool roles.
    """

    @override
    async def agenerate(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: NonNegativeInt = 128,
        logprobs: bool = False,
        top_logprobs: PositiveInt | None = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerateOutput:
        """Override agenerate to bypass validation and support tool calls."""

        if isinstance(input, str):
            return await self._generate_completion(
                input=input,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                echo=echo,
                top_logprobs=top_logprobs,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )

        return await self._generate_chat_completion(
            input=input,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            extra_body=extra_body,
        )

    def _generations_from_openai_completion_a3b(self, completion: "OpenAIChatCompletion") -> "GenerateOutput":
        """Get the generations from the OpenAI Chat Completion object.

        Args:
            completion: the completion object to get the generations from.

        Returns:
            A list of strings containing the generated responses for the input.
        """
        generations = []
        logprobs = []
        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using OpenAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
            if choice_logprobs := self._get_logprobs_from_chat_completion_choice(choice):
                logprobs.append(choice_logprobs)

        statistics = self._get_llm_statistics(completion)
        return prepare_output(
            generations=generations,
            input_tokens=statistics["input_tokens"],
            output_tokens=statistics["output_tokens"],
            logprobs=logprobs,
        )

    def _generations_from_openai_completion(self, completion: "OpenAIChatCompletion") -> "GenerateOutput":
        """Get the generations from the OpenAI Chat Completion object.

        Args:
            completion: the completion object to get the generations from.

        Returns:
            A list of strings containing the generated responses for the input.
        """
        generations = []
        logprobs = []
        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using OpenAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            if (reasoning_content := choice.message.reasoning_content) is not None:
                content = THINK_OPEN + reasoning_content.strip() + THINK_CLOSE + (content or "")
            else:
                content = THINK_OPEN + "\n\n" + THINK_CLOSE + (content or "")
            generations.append(content)
            if choice_logprobs := self._get_logprobs_from_chat_completion_choice(choice):
                logprobs.append(choice_logprobs)

        statistics = self._get_llm_statistics(completion)
        return prepare_output(
            generations=generations,
            input_tokens=statistics["input_tokens"],
            output_tokens=statistics["output_tokens"],
            logprobs=logprobs,
        )


def get_openai_client(
    model: str,
    base_url: str,
    model_type: str,
    timeout: int = 900,
    retries: int = 3,
    max_new_tokens: int = 8192,
    deterministic: bool = True,
    structured_output: dict[str, Any] | None = None,
) -> OpenAILLM:
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    params: ModelParameters = ModelType(model_type).get_model_parameters()
    generation_kwargs = params.set_recommended_defaults(generation_kwargs, deterministic=deterministic)

    return CustomOpenAILLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr("not-used"),
        timeout=timeout,
        max_retries=retries,
        generation_kwargs=generation_kwargs,
        structured_output=structured_output,
    )


def create_llm_clients(
    server: LlamaCppServerConfig | VllmServerConfig, args: DistillationConfig, schema: type[BaseModel]
) -> OpenAILLM:
    """Create structured and non-structured LLM clients."""
    base_params: dict[str, Any] = {
        "model": server.model,
        "base_url": f"http://{server.host}:{server.port}/v1",
        "timeout": args.timeout,
        "retries": args.retries,
        "max_new_tokens": args.max_new_tokens,
        "model_type": args.model_type,
        "deterministic": args.deterministic,
    }
    if args.structured_output:
        base_params["structured_output"] = {"schema": schema}
    else:
        base_params["structured_output"] = None

    llm = get_openai_client(**base_params)

    return llm


def get_function_schema() -> str:
    """Returns the tools for function calling."""
    libpath_module = load_module_from_path(get_libpath())
    tools = libpath_module.get_tools()

    function_definitions = [tool_info["function"] for tool_info in tools]
    function_schema = json.dumps(function_definitions, indent=2)

    return function_schema


def is_openai_format(messages: Any) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Turn on the living room lights."},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {
                "name": "control_light",
                "arguments": {"room": "living room", "state": "on"}
            }}]
        },
        {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
        {"role": "assistant", "content": "Done!"}
    ]
    ```

    Args:
        input: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(messages, list):
        return False
    return all(isinstance(x, dict) and "role" in x and ("content" in x or "tool_calls" in x) for x in messages)


def save_distiset_to_disk(distiset: Distiset, path: str) -> None:
    """Save the distiset to a directory."""
    distiset.save_to_disk(path)


def print_statistics(distilabel_train: list[dict[str, Any]]) -> None:
    total_train = len(distilabel_train)
    train_correct = sum(1 for row in distilabel_train if row["is_correct"])
    logger.info(f"  Math verify successes: {train_correct}/{total_train}")


def cleanup() -> None:
    """Cleans up logging to prevent multiprocessing queue errors."""
    root_logger = stdlib_logging.getLogger()
    queue_handlers = [h for h in root_logger.handlers if hasattr(h, "queue")]
    for handler in queue_handlers:
        root_logger.removeHandler(handler)

    # Reinitialize logging
    setup_logging(level=logging.INFO, include_timestamp=True)


def create_argilla_dataset_settings() -> rg.Settings:
    """Create Argilla dataset settings for linear algebra distillation results."""

    return rg.Settings(
        guidelines="""Review and validate the model's reasoning for linear algebra problems.""",
        fields=[
            rg.TextField(
                name="problem_type",
                title="Problem Type",
                use_markdown=False,
            ),
            rg.TextField(
                name="tool_calls",
                title="Number of Tool Calls Made",
                use_markdown=False,
            ),
            rg.TextField(
                name="query",
                title="User's Linear Algebra Problem Query",
                use_markdown=False,
            ),
            rg.TextField(
                name="is_correct",
                title="Is Answer Correct?",
                use_markdown=False,
            ),
            rg.TextField(
                name="ground_truth",
                title="Ground Truth Result",
                use_markdown=False,
            ),
            rg.TextField(
                name="stepwise_ground_truths",
                title="Stepwise Ground Truth Solutions",
                use_markdown=False,
            ),
            rg.TextField(
                name="final_answer",
                title="Model's Final Answer",
                use_markdown=False,
            ),
            rg.TextField(
                name="messages",
                title="Full Conversation",
                use_markdown=False,
            ),
            rg.TextField(
                name="diagnostics",
                title="Diagnostics (per turn)",
                use_markdown=False,
            ),
            rg.TextField(
                name="diagnostic_messages",
                title="Diagnostic raw messages (failed turns)",
                use_markdown=False,
            ),
            rg.TextField(
                name="composition_dependencies",
                title="Composition Dependencies",
                use_markdown=False,
            ),
            rg.TextField(
                name="composition_type",
                title="Composition Type",
                use_markdown=False,
            ),
            rg.TextField(
                name="dependency_edges",
                title="Dependency Edges",
                use_markdown=False,
            ),
            rg.TextField(
                name="model_name",
                title="Model Name Used",
                use_markdown=False,
            ),
        ],
        questions=[
            rg.LabelQuestion(
                name="reasoning_quality",
                title="How would you rate the overall reasoning quality?",
                labels=["excellent", "good", "fair", "poor"],
            ),
            rg.LabelQuestion(
                name="mathematical_accuracy",
                title="Is the mathematical reasoning correct?",
                labels=["correct", "minor_errors", "major_errors", "incorrect"],
            ),
            rg.LabelQuestion(
                name="tool_usage",
                title="Are the tool calls appropriate and effective?",
                labels=["optimal", "good", "suboptimal", "incorrect"],
            ),
            rg.LabelQuestion(
                name="final_correctness",
                title="Is the final answer correct?",
                labels=["correct", "close", "wrong", "no_answer"],
            ),
            rg.TextQuestion(
                name="feedback",
                title="Additional feedback or observations",
            ),
        ],
    )


def _format_value(value: Any) -> Any:
    """Recursively format values, applying safe_str_with_xml to strings and recursing through dicts."""
    if isinstance(value, dict):
        return {k: _format_value(v) for k, v in value.items()}
    else:
        return safe_str_with_xml(value)


def _format_indexed_list(items: list[Any]) -> str:
    """Format a list with indexed headers and separators for better readability."""
    if not items:
        return ""

    indexed_dict = []
    for i, item in enumerate(items):
        # Skip empty or whitespace-only strings
        if isinstance(item, str) and not item.strip():
            continue
        if isinstance(item, dict):
            indexed_dict.append({"index": i, "content": _format_value(item)})
        else:
            indexed_dict.append({"index": i, "content": safe_str_with_xml(item)})

    return json.dumps(indexed_dict, indent=2)


def safe_str_with_xml(value: Any) -> str:
    if value is None:
        return "N/A"
    str_value = str(value)
    return html.escape(str_value)


def _convert_item_to_argilla_record(item: dict[str, Any]) -> dict[str, str] | None:
    """Convert a single distillation item to an Argilla record."""
    logger = get_logger(__name__)
    try:
        metadata = item.get("distilabel_metadata", {})
        diagnostics_key = next((k for k in metadata if k.startswith("diagnostics_")), None)
        diagnostic_msgs_key = next((k for k in metadata if k.startswith("diagnostic_messages_")), None)
        diagnostics_list = metadata.get(diagnostics_key, []) if diagnostics_key else []
        diagnostic_msgs_list = metadata.get(diagnostic_msgs_key, []) if diagnostic_msgs_key else []

        query = item.get("query", "N/A")
        ground_truth = item.get("ground_truth", "N/A")
        stepwise_ground_truths = item.get("stepwise_ground_truths", "N/A")
        problem_type = item.get("problem_type", "N/A")
        composition_type = item.get("composition_type", "N/A")
        composition_dependencies = item.get("composition_dependencies", "N/A")
        messages = item.get("messages", [])
        dependency_edges = item.get("dependency_edges", "N/A")
        final_answer = item.get("final_answer", "N/A")
        is_correct = item.get("is_correct", "N/A")
        model_name = item.get("model_name", "N/A")
        num_tool_calls = len(json.loads(stepwise_ground_truths))

        return {
            "query": str(query),
            "ground_truth": str(ground_truth),
            "stepwise_ground_truths": str(stepwise_ground_truths),
            "tool_calls": str(num_tool_calls),
            "problem_type": str(problem_type),
            "composition_type": str(composition_type),
            "composition_dependencies": str(composition_dependencies),
            "messages": _format_indexed_list(messages),
            "dependency_edges": str(dependency_edges),
            "final_answer": str(final_answer),
            "is_correct": str(is_correct),
            "model_name": str(model_name),
            "diagnostics": _format_indexed_list(diagnostics_list),
            "diagnostic_messages": _format_indexed_list(diagnostic_msgs_list),
        }
    except Exception as e:
        logger.warning(f"Failed to process record: {e}")
        return None


def create_argilla_dataset(
    dataset_name: str, distiset_data: list[dict[str, Any]], client: rg.Argilla, private: bool
) -> None:
    """Create and populate an Argilla dataset from distillation results."""
    logger = get_logger(__name__)

    try:
        # Create dataset with settings
        settings = create_argilla_dataset_settings()
        dataset = rg.Dataset(
            name=dataset_name,
            settings=settings,
            client=client,
        )
        _ = dataset.create()
        logger.info(f"Created Argilla dataset: {dataset_name}")

        # Convert distilabel data to Argilla records
        records = []
        for item in distiset_data:
            record = _convert_item_to_argilla_record(item)
            if record is not None:
                records.append(record)

        # Log records to dataset
        if records:
            dataset.records.log(records=records)
            logger.info(f"Logged {len(records)} records to Argilla dataset")
        else:
            logger.warning("No valid records found to log")
        domain = dataset_name.replace("/", "-").replace("-debug", "").replace("-train", "").replace("-validation", "")
        logger.info("✅ Argilla dataset created successfully")
        logger.info(f"   Privacy: {'Private' if private else 'Public'}")
        logger.info(f"   Access URL: https://{domain}.hf.space")
    except Exception:
        logger.exception("Failed to create Argilla dataset")


def filter_dataset_by_correctness(distiset: Distiset, is_correct: bool = True) -> Distiset:
    """Filter dataset by is_correct flag."""

    filtered_distiset = deepcopy(distiset)

    for split_name in filtered_distiset["default"]:
        split_data = filtered_distiset["default"][split_name]
        # Keep only correct entries for SFT training if only_correct=True, otherwise keep all
        filtered_data = split_data.filter(lambda x: x["is_correct"] is is_correct)

        filtered_distiset["default"][split_name] = filtered_data

    return filtered_distiset


def push_to_huggingface(distiset: Distiset, dataset_name: str, private: bool) -> None:
    prepare_dataset_for_sft(distiset)
    strip_diagnostic_messages_from_metadata(distiset)
    normalize_schema(distiset)

    try:
        distiset.push_to_hub(
            dataset_name,
            private=private,
        )
        logger.info(f"✅ Dataset successfully pushed to: {dataset_name}")
        logger.info(f"   Privacy: {'Private' if private else 'Public'}")
        logger.info(f"   Access URL: https://huggingface.co/datasets/{dataset_name}")
    except Exception:
        logger.exception("Failed to push dataset to Hugging Face Hub")


def push_argilla_dataset(argilla_client: rg.Argilla, distiset: Distiset, args: DistillationConfig) -> None:
    success = filter_dataset_by_correctness(distiset, is_correct=True)
    if len(success["default"]["train"]) > 0:
        create_argilla_dataset(
            dataset_name=f"{args.argilla_output_dataset}",
            distiset_data=success["default"]["train"],
            client=argilla_client,
            private=args.private,
        )
    failures = filter_dataset_by_correctness(distiset, is_correct=False)
    if len(failures["default"]["train"]) > 0:
        create_argilla_dataset(
            dataset_name=f"{args.argilla_output_dataset}-failures",
            distiset_data=failures["default"]["train"],
            client=argilla_client,
            private=args.private,
        )


def push_datasets_to_huggingface(distiset: Distiset, args: DistillationConfig) -> None:
    """Push two datasets to Hugging Face: one with all entries and one with only correct entries."""
    assert args.hf_output_dataset is not None
    private = args.private

    # Push all entries dataset
    all_entries_name = f"{args.hf_output_dataset}-failures"
    logger.info(f"Pushing dataset with all entries to: {all_entries_name}")
    all_entries_distiset = filter_dataset_by_correctness(distiset, is_correct=False)
    if len(all_entries_distiset["default"]["train"]) > 0:
        push_to_huggingface(all_entries_distiset, all_entries_name, private)

    # Push correct entries only dataset
    correct_only_name = args.hf_output_dataset
    logger.info(f"Pushing dataset with correct entries only to: {correct_only_name}")
    correct_only_distiset = filter_dataset_by_correctness(distiset, is_correct=True)
    if len(correct_only_distiset["default"]["train"]) > 0:
        push_to_huggingface(correct_only_distiset, correct_only_name, private)


def prepare_dataset_for_sft(distiset: Distiset) -> None:
    """Adds the tools column to the dataset."""
    TOOLS = get_tools()

    def add_tools_column(example: dict[str, Any]) -> dict[str, Any]:
        example["tools"] = TOOLS
        return example

    distiset["default"]["train"] = distiset["default"]["train"].map(add_tools_column)
    if "validation" in distiset["default"]:
        distiset["default"]["validation"] = distiset["default"]["validation"].map(add_tools_column)


def normalize_schema(distiset: Distiset) -> None:
    ns = distiset["default"]

    # 1) Stringify nested columns if present
    for split in list(ns.keys()):
        if "messages" in ns[split].column_names:
            ns[split] = ns[split].map(lambda r: {"messages": json.dumps(r.get("messages", []))})
        if "distilabel_metadata" in ns[split].column_names:
            ns[split] = ns[split].map(lambda r: {"distilabel_metadata": json.dumps(r.get("distilabel_metadata", {}))})

    # 2) Align columns by UNION: add missing columns with empty placeholders
    all_cols = set()
    for split in ns:
        all_cols |= set(ns[split].column_names)

    for split in list(ns.keys()):
        missing = sorted(all_cols - set(ns[split].column_names))
        if missing:
            for col in missing:
                ns[split] = ns[split].add_column(col, [None] * len(ns[split]))


def convert_dataset_to_list_of_dicts(dataset: Dataset) -> list[dict[str, Any]]:
    """Convert dataset from dict format to list of dicts."""
    dataset_dict = dataset.to_dict()
    return [dict(zip(dataset_dict.keys(), vals, strict=True)) for vals in zip(*dataset_dict.values(), strict=True)]


def load_dataset_split(
    dataset_name: str, dataset_config: str | None, split: str, take_n: int | None = None
) -> Dataset:
    """Loads a single dataset split either from the hub or from a local file."""
    logger = get_logger(__name__)

    try:
        logger.info(f"Loading '{dataset_name}' (config: {dataset_config}, split: {split}) dataset.")

        dataset = hf_load_dataset(dataset_name, dataset_config, split=split)
        assert isinstance(dataset, Dataset)

        logger.info("Dataset loaded!")
    except Exception as err:
        raise FileNotFoundError(f"The dataset {dataset_name} is not available on the Hugging Face Hub.") from err
    else:
        if take_n is not None:
            dataset = dataset.select(range(take_n))
        return dataset


def load_datasets_for_distillation(args: DistillationConfig) -> dict[str, list[dict[str, Any]]]:
    """Loads train and optionally validation splits as lists of dicts."""
    take_n = args.take_n
    datasets: dict[str, list[dict[str, Any]]] = {}
    if args.dataset_name is None:
        raise ValueError("dataset_name must be provided")

    dataset = load_dataset_split(args.dataset_name, args.dataset_config, "train", take_n=take_n)
    datasets["train"] = convert_dataset_to_list_of_dicts(dataset)

    return datasets


def strip_diagnostic_messages_from_metadata(distiset: Distiset) -> None:
    """Remove diagnostic_messages_* keys from distilabel_metadata for all splits (before HF push)."""
    ns = distiset["default"]

    def strip_md(record: dict[str, Any]) -> dict[str, Any]:
        md = record.get("distilabel_metadata", {})
        if isinstance(md, dict):
            keys_to_remove = [k for k in md if k.startswith("diagnostic_messages_")]
            if keys_to_remove:
                for k in keys_to_remove:
                    md.pop(k, None)
                return {"distilabel_metadata": md}
        return {"distilabel_metadata": md}

    for split in list(ns.keys()):
        if "distilabel_metadata" in ns[split].column_names:
            ns[split] = ns[split].map(strip_md)
