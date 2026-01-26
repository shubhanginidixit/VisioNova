import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Standard text preprocessing for NLP tasks.
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            logger.warning("Stopwords not found. Downloading...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
        if self.lemmatize:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Test the lemmatizer by calling it once
                self.lemmatizer.lemmatize('test')
            except Exception:
                logger.warning("WordNet not found. Downloading...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                nltk.download('wordnet_ic', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Basic cleaning: lowercase, remove punctuation, remove extra whitespace.
        """
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove newlines and extra spaces
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        """
        text = self.clean_text(text)
        
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]
            
        if self.lemmatize and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
            except Exception as e:
                logger.warning(f"Lemmatization failed: {e}. Using tokens as-is.")
            
        return " ".join(tokens)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        """Applies preprocessing to a pandas Series."""
        return series.apply(self.preprocess)
