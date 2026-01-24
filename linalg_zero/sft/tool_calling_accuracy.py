"""
Tool calling accuracy callback for SFT training.

Evaluates structural and correctnessfo metrics for tool-use generations on all eval data.
"""

from __future__ import annotations

import json as _json
import os
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from linalg_zero.distillation.components.models import DefaultConfig
from linalg_zero.distillation.data import FunctionInvocationInfo, ThoughtSchema
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.sft.diagnostics import DiagnosticTracker
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.lib import get_lib, get_lib_fn_names
from linalg_zero.shared.system_prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
)
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class ToolCallingAccuracyCallback(TrainerCallback):
    """
    Callback to evaluate tool calling accuracy during SFT training.

    Evaluates all samples in the eval_dataset for robust metric computation.
    """

    def __init__(self, model_name: str, dataset_name: str, eval_dataset: Any) -> None:
        self.eval_dataset = eval_dataset
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.library = get_lib()
        self._parser = XMLParser()
        self.model_config = DefaultConfig()
        self.generation_config = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")

        if model is None or tokenizer is None:
            return

        # Get max_new_tokens from training args (eval_max_new_tokens config)
        max_new_tokens = getattr(args, "eval_max_new_tokens", 1024)

        # Determine eval subset size
        max_eval_samples = getattr(args, "max_eval_samples", None)
        if max_eval_samples is None or max_eval_samples < 0:
            num_samples = len(self.eval_dataset)
        else:
            num_samples = min(max_eval_samples, len(self.eval_dataset))

        logger.info(f"Computing tool-calling metrics on {num_samples}/{len(self.eval_dataset)} eval samples...")

        all_messages, weave_metadata, eval_metrics = self._compute_metrics(
            model=model,
            tokenizer=tokenizer,
            dataset=self.eval_dataset,
            max_new_tokens=max_new_tokens,
            max_samples=num_samples,
        )
        if eval_metrics:
            self._log_evaluation_metrics(eval_metrics, state, prefix="eval")
            if metrics is not None:
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        self._log_to_weave(state, all_messages, weave_metadata)

    def _log_evaluation_metrics(self, metrics: dict[str, float], state: TrainerState, prefix: str = "eval") -> None:
        """Log evaluation metrics to trainer state and logger (Trainer will forward to W&B)."""
        for metric_name, value in metrics.items():
            logger.info(f"eval/{metric_name}: {value:.3f}")

    def _log_to_weave(
        self, state: TrainerState, all_messages: list[list[dict[str, Any]]], metadata: dict[str, int]
    ) -> None:
        """Log predictions and summary to Weave."""
        # Imported lazily: weave is a training-only dependency, kept out of the import path
        # so the pure scoring/extraction methods stay importable without it.
        from weave import EvaluationLogger
        from weave.trace.context import weave_client_context

        client = weave_client_context.get_weave_client()
        if client is not None:
            eval_attributes = {
                "training_step": state.global_step,
                "model_name": self.model_name,
                "generation_config": (self.generation_config if self.generation_config else None),
            }

            # Sanitize model name for Weave scorer validation
            sanitized_model_name = self.model_name.replace("/", "_").replace("-", "_").replace(".", "_")

            weave_logger = EvaluationLogger(
                name=f"linalg_zero_sft_eval_{state.global_step}",
                scorers=[],
                model=sanitized_model_name,
                dataset=self.dataset_name,
                eval_attributes=eval_attributes,
            )
        else:
            return

        logger.info(f"Logging {len(all_messages)} samples to Weave...")

        # Log each sample as a prediction
        for messages in all_messages:
            inputs = {"messages": messages[:2]}
            output = {"messages": messages[2:]} if len(messages) > 2 else None
            pred = weave_logger.log_prediction(inputs=inputs, output=output)
            pred.finish()

        # Log summary with metadata
        weave_logger.log_summary(metadata)
        logger.info(f"Weave logging completed with metadata: {metadata}")

    def _compute_metrics(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Any,
        max_new_tokens: int,
        max_samples: int | None = None,
    ) -> tuple[list[list[dict[str, Any]]], dict[str, int | float], dict[str, float]]:
        """Compute metrics on eval samples with fair turn allocation per sample."""
        model.eval()

        if not dataset or len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Subset dataset if max_samples specified
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        samples: list[dict[str, Any]] = [dataset[i] for i in range(num_samples)]

        # Initialize metrics tracker
        tracker = DiagnosticTracker()

        pbar = tqdm(samples, desc="Evaluating tool calling", unit="sample", disable=False)
        for sample in pbar:
            # Determine fair n_turns based on sample complexity
            steps = sample["stepwise_ground_truths"]
            arr = _json.loads(steps)
            num_tool_turns = len(arr)
            n_turns = num_tool_turns + 1

            # Run evaluation and update tracker
            state = self._run_evaluation_turns(model, tokenizer, sample, n_turns, max_new_tokens)
            tracker.update(state)

            # Update progress bar
            pbar.set_postfix({**tracker.get_progress_info()})

        all_messages, metadata, eval_metrics = tracker.get_history()
        return all_messages, metadata, eval_metrics

    def _generate(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_text: str, max_new_tokens: int
    ) -> str:
        """Generate output for a single prompt."""
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            padding=bool(getattr(tokenizer, "pad_token_id", None)),
        )
        if inputs["input_ids"].shape[1] == tokenizer.model_max_length:
            logger.warning(f"Input truncated to {tokenizer.model_max_length} tokens during tool calling evaluation")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.generation_config is None:
                self.generation_config = self.get_generation_config(max_new_tokens, tokenizer)

            outputs = model.generate(  # type: ignore[operator]
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                tokenizer=tokenizer,
                **self.generation_config,
            )

        # Extract only the generated tokens (after the input)
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]

        # Special tokens to preserve
        KEEP_TOKENS = {ANSWER_OPEN, ANSWER_CLOSE, THINK_OPEN, THINK_CLOSE, TOOL_CALL_OPEN, TOOL_CALL_CLOSE}

        # Decode without skipping any special tokens
        output_raw = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)

        # Remove unwanted special tokens
        output = output_raw
        for special_token in tokenizer.all_special_tokens:
            if special_token not in KEEP_TOKENS:
                output = output.replace(special_token, "")

        # Check if generation was truncated due to max_new_tokens
        if (
            generated_tokens.shape[1] == max_new_tokens
            and getattr(tokenizer, "eos_token_id", None) is not None
            and generated_tokens[0, -1].item() != tokenizer.eos_token_id
        ):
            logger.warning(f"Generation may have been truncated at max_new_tokens={max_new_tokens}")

        if os.getenv("SFT_LOG_GENERATIONS") == "1":
            logger.info("=" * 100)
            logger.info(f"Generated output (len={len(output)}): {output}")
            logger.info("=" * 100)

        return output

    def get_generation_config(self, max_new_tokens: int, tokenizer: PreTrainedTokenizer) -> dict[str, Any]:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        assert eos_id is not None, "EOS token ID is not set"
        assert pad_id is not None, "PAD token ID is not set"

        stop_strings = [TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE]
        stop_token_ids = [eos_id]
        for token_str in [TOOL_CALL_CLOSE, ANSWER_CLOSE]:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) != 1:
                logger.warning(
                    f"Special token '{token_str}' was split into {len(token_ids)} tokens. "
                    f"Ensure it's registered as a special token."
                )
            else:
                stop_token_ids.append(token_ids[0])

        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": pad_id,
            "eos_token_id": stop_token_ids,
            "stop_strings": stop_strings,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "use_cache": True,
        }

    def _run_evaluation_turns(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample: dict[str, Any],
        n_turns: int,
        max_new_tokens: int,
    ) -> EvaluationState:
        """Run evaluation using simplified GRPO-based conversation processing."""
        state = EvaluationState()
        state.sample = sample  # Store the original dataset sample

        context = list(sample["messages"])
        all_context = list(sample["messages"])
        tools = sample["tools"]

        # Multi-turn conversation loop
        for _ in range(n_turns):
            # Generate assistant response (include tools so template exposes function signatures)
            prompt = tokenizer.apply_chat_template(context, tools=tools, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                state.early_stop_reason = "prompt_format_error"
                break

            output = self._generate(model, tokenizer, prompt, max_new_tokens)
            message = self.extract_exact_match(output, context)
            self.add_message("assistant", all_context, output, unstructured=True)

            # Check if message extraction worked
            if message is None:
                state.early_stop_reason = "extraction_failed"
                break

            if message.final_answer is None and message.tool_call is None:
                state.early_stop_reason = "no_action"
                break

            # Track diagnostic info
            if message.tool_call is not None:
                state.tool_parse_success = True

            self.add_message("assistant", context, message)

            # Execute tool call if it exists
            if message.tool_call is not None:
                tool_call = self._execute(message)
                self.add_message("tool", context, tool_call)
                self.add_message("tool", all_context, tool_call, unstructured=True)

            if message.final_answer is not None:
                state.generated_answer = message.final_answer
                state.early_stop_reason = "final_answer_provided"
                break

        # Compute partial format score on complete conversation
        state.strict_format_match = self.calculate_exact_match(context)
        state.partial_format_score = self.calculate_partial_match(all_context)
        state.messages = all_context

        return state

    def _execute(self, msg: ThoughtSchema) -> dict[str, str]:
        if msg.tool_call is None:
            raise ValueError("Tool call is required for execution")

        name = msg.tool_call.name
        arguments = msg.tool_call.arguments

        try:
            if name not in self.library:
                return {
                    "function_name": name,
                    "execution_result": f"ERROR: Function '{name}' not found in library",
                }

            result = self.library[name](**arguments)
            return {"function_name": name, "execution_result": str(result)}
        except Exception as exc:
            return {
                "function_name": name,
                "execution_result": f"ERROR: {type(exc).__name__}: {exc}",
            }

    def calculate_exact_match(self, context: list[dict]) -> float:
        """Calculate exact match for the entire conversation."""
        reward: list[float] = []

        # Discard the first two turns (system and user)
        for completion in context[2:]:
            msg = completion["content"]
            score = 1 if self._parser._is_valid_think_then_tool_or_answer(msg) else 0
            reward.append(score)

        return sum(reward) / len(reward) if len(reward) > 0 else 0

    def calculate_partial_match(self, context: list[dict]) -> float:
        """
        Compute format adherence score for lenient evaluation.
        """
        scores: list[float] = []

        # Discard the first two turns (system and user)
        for completion in context[2:]:
            score: float = 0
            tool_or_answer_reward = 0
            response = completion["content"]

            # No need to reward <start_working_out> since we always prepend it
            # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 1 if response.count(THINK_CLOSE) == 1 else 0
            tool_or_answer_reward += 1 if response.count(TOOL_CALL_OPEN) == 1 else 0
            tool_or_answer_reward += 1 if response.count(TOOL_CALL_CLOSE) == 1 else 0

            if tool_or_answer_reward == 0:
                tool_or_answer_reward += 1 if response.count(ANSWER_OPEN) == 1 else 0
                tool_or_answer_reward += 1 if response.count(ANSWER_CLOSE) == 1 else 0

            score += tool_or_answer_reward
            scores.append(score)

        return sum(scores) / len(scores) if len(scores) > 0 else 0

    def extract_exact_match(self, message: str, context: list[dict]) -> ThoughtSchema | None:
        """Extract output from messages that do not enforce structured output."""
        analysis = self._parser.analyze_message_in_context(context, message=message, tool_names=get_lib_fn_names())

        if not bool(analysis["is_valid_think_then_tool_or_answer"]):
            return None

        if analysis["has_answer"] and not bool(analysis["answer_policy_valid"]):
            return None

        thought = analysis["thought"] or ""

        # Enforce a single tool call per turn: take only the last tool block
        tool_call: FunctionInvocationInfo | None = None
        tool_info = analysis["tool"]
        if tool_info and tool_info["json_valid"]:
            tool_call = FunctionInvocationInfo(
                name=str(tool_info["name"]),
                arguments=dict(tool_info["arguments"]),
            )

        # Mark completion based on presence of answer
        answer = analysis["answer"]
        return ThoughtSchema(
            thought=thought,
            tool_call=tool_call,
            final_answer=answer,
            completed=answer is not None,
        )

    def add_message(
        self,
        role: str,
        context: list[dict[str, Any]],
        message: ThoughtSchema | dict[str, str] | str,
        *,
        unstructured: bool = False,
    ) -> None:
        if role == "assistant":
            if isinstance(message, str):
                msg = {"role": "assistant", "content": message}
            else:
                assert isinstance(message, ThoughtSchema)
                msg = self.model_config.format_assistant_message(message)
        elif role == "tool":
            if unstructured:
                assert isinstance(message, dict)
                msg = {
                    "role": "tool",
                    "content": message["execution_result"],
                    "name": message["function_name"],
                }
            else:
                assert isinstance(message, dict)
                msg = self.model_config.create_tool_message(context, message)
        else:
            raise ValueError(f"Invalid role: {role}")

        assert msg is not None, f"Message is None for role: {role}"
        context.append(msg)
