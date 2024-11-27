import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sentiment_analyzer import (
    extract_content_from_website,
    find_similar_words,
    sentiment_analysis
)

# Mock URL and HTML content for testing
mock_html = """
<html>
    <head><title>Test Page</title></head>
    <body>
        <p>This is a sample page about Artificial Intelligence.</p>
        <p>It discusses AI and Machine Learning concepts.</p>
    </body>
</html>
"""

@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mock the requests.get function to return test HTML content."""
    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.content = text.encode("utf-8")
            self.status_code = status_code

    def mock_get(url, headers=None):
        return MockResponse(mock_html, 200)

    monkeypatch.setattr("requests.get", mock_get)


def test_extract_content_from_website(mock_requests_get):
    """Test if the content extraction from a website works as expected."""
    url = "https://example.com"
    content = extract_content_from_website(url)
    assert "Artificial Intelligence" in content
    assert "Machine Learning" in content
    assert "Test Page" in content


def test_find_similar_words():
    """Test if similar words are retrieved correctly using WordNet."""
    word = "AI"
    similar_words = find_similar_words(word)
    assert "AI" in similar_words or similar_words  # Check for synonyms
    assert isinstance(similar_words, set)  # Ensure the result is a set


def test_sentiment_analysis():
    """Test if sentiment analysis works as expected."""
    positive_text = "This is an amazing product with excellent features."
    negative_text = "This is a terrible product and a complete waste of money."
    neutral_text = "This is a product."

    assert sentiment_analysis(positive_text) == "Positive"
    assert sentiment_analysis(negative_text) == "Negative"
    assert sentiment_analysis(neutral_text) == "Neutral"
