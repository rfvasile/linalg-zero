import pytest

from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback


@pytest.fixture
def callback() -> ToolCallingAccuracyCallback:
    """A callback for exercising the pure scoring/extraction methods.

    Construction only touches local objects (get_lib, XMLParser, DefaultConfig); the
    methods under test never read eval_dataset, the model, or the tokenizer.
    """
    return ToolCallingAccuracyCallback(model_name="test", dataset_name="test", eval_dataset=None)
