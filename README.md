# **Content Sentiment Analyzer** 📊

This Python-based tool extracts and analyzes text content from web pages and transcripts, identifying sentiment for specific keywords. It uses web scraping, text processing, and sentiment analysis to provide insights into keyword sentiment trends.

---

## 🚀 **Features**
- **Content Extraction**: Extracts textual content from web pages using HTTP requests and BeautifulSoup.
- **Transcript Support**: Handles audio/video transcript files using `speech_recognition`.
- **Text Processing**:
  - Tokenizes and lemmatizes text for meaningful word analysis.
  - Finds similar words and synonyms using WordNet.
- **Keyword Analysis**:
  - Tracks keyword occurrences and performs threshold-based analysis.
- **Sentiment Analysis**: Determines if content sentiment is `Positive`, `Negative`, or `Neutral` using TextBlob.
- **Results Display**: Outputs results in a pandas DataFrame for easy readability.

---

## 📂 **Project Structure**
```plaintext
content-sentiment-analyzer/
├── src/
│   ├── sentiment_analyzer.py  # Main script for content analysis
├── test/
│   ├── test_analyzer.py             # Example transcript for testing (optional)
├── requirements.txt                  # List of dependencies
├── README.md                         # Documentation
├── .gitignore                        # Ignored files
```

---

## 🛠️ Installation

Follow these steps to set up and run the project locally:

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/content-sentiment-analyzer.git
cd content-sentiment-analyzer

```

### Step 2: Install Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Resources
Ensure necessary NLTK resources are downloaded:
```bash
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### Step 4: Run the Analyzer
Execute the script to start analyzing:
```bash
python src/sentiment_analyzer.py
```


## ⚙️ Configuration

You can customize the following settings in `src/sentiment_analyzer.py` to tailor the analyzer to your needs:

- **Update the `content_sources` list** in the script with your desired URLs or file paths:
   ```python
   content_sources = [
       "https://example.com/article1",
       "https://example.com/article2",
   ]
- **Modify the `keywords` list** to include the keywords you want to analyze:
  ```python
  keywords = ["AI", "Machine Learning", "Artificial Intelligence"]
  

## 🛠️ Technologies Used

The project leverages the following technologies:

- **Python**: Core programming language for logic implementation.
- **NLTK**: Natural Language Toolkit for tokenization, lemmatization, and synonym detection.
- **TextBlob**: Sentiment analysis for text polarity determination.
- **BeautifulSoup**: Web scraping for extracting content from HTML pages.
- **pandas**: Data manipulation and visualization of results.


## 📊 Example Output

Below is an example of the output generated by the script:

The script processes the content and produces a DataFrame with the following columns:
- **Content Source**: The source URL or file path.
- **Keyword**: The keyword analyzed.
- **Sentiment**: Sentiment for the content (`Positive`, `Negative`, `Neutral`).

### Sample DataFrame:
| Content Source                                                | Keyword | Sentiment  |
|---------------------------------------------------------------|---------|------------|
| https://venturebeat.com/ai/here-is-how-far-we-are-to-achieving-agi | agi     | Positive   |
| https://www.theverge.com/2024/1/18/24042354/mark-zuckerberg-meta-agi-reorg-interview | agi | Negative   |
