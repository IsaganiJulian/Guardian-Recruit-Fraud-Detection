"""
NLP Stream Module for Guardian Recruit Fraud Detection
Uses BERT/RoBERTa for text analysis of job postings
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class NLPStream:
    """
    NLP Stream for analyzing job posting text using BERT/RoBERTa models
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the NLP stream with a pre-trained transformer model
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2  # Binary classification: fraud/legitimate
        )
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, text):
        """
        Extract features from text using the transformer model
        
        Args:
            text (str): Job posting text to analyze
            
        Returns:
            np.ndarray: Feature embeddings
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        # Tokenize input text
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Extract embeddings from the last hidden state
        embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        
        return embeddings
    
    def analyze_text(self, text):
        """
        Analyze job posting text for fraud indicators
        
        Args:
            text (str): Job posting text
            
        Returns:
            dict: Analysis results including features and confidence scores
        """
        features = self.extract_features(text)
        
        return {
            'features': features,
            'feature_dim': features.shape[1]
        }
