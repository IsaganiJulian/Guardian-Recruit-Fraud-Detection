"""
NLP Stream Module - BERT/RoBERTa Text Analysis
Handles natural language processing for job posting fraud detection using transformer models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class NLPStream:
    """
    Text analysis stream using BERT/RoBERTa for detecting fraudulent job postings.
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        """
        Initialize the NLP stream with a pre-trained transformer model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2  # Binary classification: fraud vs legitimate
        )
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, texts):
        """
        Extract features from text using the transformer model.
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            np.ndarray: Extracted features for each text
        """
        if self.model is None:
            self.load_model()
            
        features = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize the text
                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Use the CLS token representation from the last hidden state
                cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                features.append(cls_embedding.flatten())
                
        return np.array(features)
    
    def predict(self, texts):
        """
        Predict fraud probability for given texts.
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            np.ndarray: Fraud probabilities for each text
        """
        if self.model is None:
            self.load_model()
            
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.append(probs[0][1].cpu().item())  # Probability of fraud class
                
        return np.array(predictions)
