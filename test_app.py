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

from unittest.mock import patch
import json

@patch("urllib.request.urlopen")
def test_get_current_weather_default_unit(mock_urlopen):
    """Test get_current_weather with default unit (celsius) using mocked Open-Meteo."""
    # Mock Geocode response
    mock_geocode_response = MagicMock()
    mock_geocode_response.read.return_value = json.dumps({
        "results": [{"latitude": 37.7749, "longitude": -122.4194}]
    }).encode('utf-8')

    # Mock Weather response
    mock_weather_response = MagicMock()
    mock_weather_response.read.return_value = json.dumps({
        "current_weather": {"temperature": 15.5, "weathercode": 2}
    }).encode('utf-8')

    mock_urlopen.side_effect = [mock_geocode_response, mock_weather_response]

    location = "San Francisco, CA"
    result = get_current_weather(location)

    expected = {"temperature": 15.5, "weather": "Partly cloudy", "unit": "celsius"}
    assert result == expected

@patch("urllib.request.urlopen")
def test_get_current_weather_fahrenheit(mock_urlopen):
    """Test get_current_weather with fahrenheit unit."""
    # Mock Geocode response
    mock_geocode_response = MagicMock()
    mock_geocode_response.read.return_value = json.dumps({
        "results": [{"latitude": 35.6895, "longitude": 139.6917}]
    }).encode('utf-8')

    # Mock Weather response
    mock_weather_response = MagicMock()
    mock_weather_response.read.return_value = json.dumps({
        "current_weather": {"temperature": 75.2, "weathercode": 1}
    }).encode('utf-8')

    mock_urlopen.side_effect = [mock_geocode_response, mock_weather_response]

    location = "Tokyo, JP"
    result = get_current_weather(location, unit="fahrenheit")

    expected = {"temperature": 75.2, "weather": "Mainly clear", "unit": "fahrenheit"}
    assert result == expected

@patch("urllib.request.urlopen")
def test_get_current_weather_location_not_found(mock_urlopen):
    """Test get_current_weather when geocoding fails to find the location."""
    mock_geocode_response = MagicMock()
    mock_geocode_response.read.return_value = json.dumps({"results": []}).encode('utf-8')
    mock_urlopen.return_value = mock_geocode_response

    location = "NowhereCityThatDoesNotExist"
    result = get_current_weather(location)

    assert "error" in result
    assert "not found" in result["error"].lower()
