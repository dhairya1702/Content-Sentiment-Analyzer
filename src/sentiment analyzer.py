import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
from textblob import TextBlob


def extract_content_from_website(url):
    """
    Extracts content from a given website URL.

    Parameters:
    url (str): The URL of the website to extract content from.

    Returns:
    str: The extracted content from the website, or None if extraction fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text_elements = soup.find_all(string=True)
            text_content = ' '.join(element.strip() for element in text_elements if element.strip())
            return text_content
        else:
            print(f"Error: Unable to fetch content from the website. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print("Error:", e)
        return None


def extract_content_from_transcript(file_path):
    """
    Extracts content from an audio or video transcript file.

    Parameters:
    file_path (str): The file path of the audio or video transcript.

    Returns:
    str: The extracted content from the transcript file, or None if extraction fails.
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_text = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_text)
        return transcript
    except Exception as e:
        print("Error:", e)
        return None


def find_similar_words(word):
    """
    Finds similar words and synonyms for a given word using WordNet.

    Parameters:
    word (str): The word to find similar words for.

    Returns:
    set: A set of similar words and synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def sentiment_analysis(text):
    """
    Performs sentiment analysis on a given text using TextBlob.

    Parameters:
    text (str): The text to analyze for sentiment.

    Returns:
    str: The sentiment analysis result ('Positive', 'Negative', 'Neutral').
    """
    try:
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        if sentiment_score > 0:
            return "Positive"
        elif sentiment_score < 0:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print("Error:", e)
        return None


# List of keywords
keywords = ["AGI", "agi", "Artificial General Intelligence"]

# List of content sources (e.g., URLs or file paths)
content_sources = ["https://venturebeat.com/ai/here-is-how-far-we-are-to-achieving-agi-according-to-deepmind/",
                   "https://www.theverge.com/2024/1/18/24042354/mark-zuckerberg-meta-agi-reorg-interview",
                   "https://www.cioreview.com/news/artificial-general-intelligence-agi-the-ultimate-goal-of-ai-research-nid-38741-cid-175.html",
                   "https://www.technologyreview.com/2023/11/16/1083498/google-deepmind-what-is-artificial-general-intelligence-agi/",
                   "https://gizmodo.com/sam-altman-openai-q-machine-learning-artificial-intelli-1851062584",
                   "https://www.engadget.com/mark-zuckerberg-is-the-latest-billionaire-who-wants-to-create-artificial-general-intelligence-210820789.html",
                   "https://www.cnet.com/tech/computing/dream-on-mark-zuckerberg-your-new-ai-bet-is-a-real-long-shot/"]

# List to store results
results = []

# Loop through each content source (websites, transcripts, etc.)
for content_source in content_sources:
    if content_source.startswith("https://"):
        content = extract_content_from_website(content_source)
    else:
        content = extract_content_from_transcript(content_source)

    if content:
        # Tokenize content
        words = word_tokenize(content.lower())

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Count occurrences of each keyword
        word_count = Counter(words)

        # Check for keyword occurrence and perform sentiment analysis
        for keyword in keywords:
            occurrences = word_count[keyword]
            if occurrences < 7:
                # Find similar words for the keyword
                similar_words = find_similar_words(keyword)
                # Calculate total occurrences of keyword + similar words
                occurrences = sum(word_count[word] for word in similar_words) + occurrences
            if occurrences > 7:
                sentiment = sentiment_analysis(content)
                results.append({"Content Source": content_source, "Keyword": keyword, "Sentiment": sentiment})
            else:
                pass

    else:
        print("Error: Unable to extract content from", content_source)

# Display results in a table
results_df = pd.DataFrame(results)
print(results_df)


