
import sys
from unittest.mock import MagicMock

# Mocking heavy dependencies before importing app
mock_transformers = MagicMock()
mock_transformers_utils = MagicMock()
mock_transformers.utils = mock_transformers_utils
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.utils"] = mock_transformers_utils

sys.modules["torch"] = MagicMock()
sys.modules["gradio"] = MagicMock()
sys.modules["ddgs"] = MagicMock()
sys.modules["yfinance"] = MagicMock()

import app

def test_extract_tool_calls_logic():
    text = '<|tool_call>call:get_current_weather{location: "Tokyo", unit: "celsius"}<tool_call|>'
    calls = app.extract_tool_calls(text)
    assert calls == [{'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}]

    text = '<|tool_call>call:test_tool{int_val: 123, float_val: 45.67, bool_val: true, str_val: "hello"}<tool_call|>'
    calls = app.extract_tool_calls(text)
    assert calls == [{'name': 'test_tool', 'arguments': {'int_val': 123, 'float_val': 45.67, 'bool_val': True, 'str_val': 'hello'}}]

def test_cast_vulnerability_fix():
    # Injected into the app's extract_tool_calls scope
    # Since cast is a nested function, we can't test it directly from the module
    # But we can test it through extract_tool_calls

    # We want to ensure it doesn't crash on non-convertible strings
    text = '<|tool_call>call:test{arg: "not_a_number"}<tool_call|>'
    calls = app.extract_tool_calls(text)
    assert calls[0]['arguments']['arg'] == "not_a_number"

    # Ensure it handles booleans
    text = '<|tool_call>call:test{arg: true}<tool_call|>'
    calls = app.extract_tool_calls(text)
    assert calls[0]['arguments']['arg'] is True
