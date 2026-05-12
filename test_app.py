import sys
from unittest.mock import MagicMock

# Mocking dependencies to avoid ModuleNotFoundError and heavy loading
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.utils"] = MagicMock()

mock_gradio = MagicMock()
sys.modules["gradio"] = mock_gradio

mock_ddgs = MagicMock()
sys.modules["ddgs"] = mock_ddgs

mock_yfinance = MagicMock()
sys.modules["yfinance"] = mock_yfinance

# Now import the function to test from app.py
from app import get_current_weather

def test_get_current_weather_default_unit():
    """Test get_current_weather with default unit (celsius)."""
    location = "San Francisco, CA"
    result = get_current_weather(location)

    expected = {"temperature": 22, "weather": "partly cloudy", "unit": "celsius"}
    assert result == expected

def test_get_current_weather_fahrenheit():
    """Test get_current_weather with fahrenheit unit."""
    location = "Tokyo, JP"
    result = get_current_weather(location, unit="fahrenheit")

    expected = {"temperature": 22, "weather": "partly cloudy", "unit": "fahrenheit"}
    assert result == expected
