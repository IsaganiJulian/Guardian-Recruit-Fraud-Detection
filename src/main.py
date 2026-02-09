"""
Main Module - System Integration
Integrates all components of the Guardian Recruit Fraud Detection system.
"""

import numpy as np
import pandas as pd
from nlp_stream import NLPStream
from outlier_stream import OutlierStream
from fusion_layer import FusionLayer


class GuardianRecruitFraudDetector:
    """
    Main fraud detection system that integrates NLP, outlier detection, and fusion layer.
    """
    
    def __init__(self, model_name='bert-base-uncased', contamination=0.1):
        """
        Initialize the fraud detection system.
        
        Args:
            model_name (str): Pre-trained transformer model name
            contamination (float): Expected proportion of outliers
        """
        self.nlp_stream = NLPStream(model_name=model_name)
        self.outlier_stream = OutlierStream(contamination=contamination)
        self.fusion_layer = FusionLayer()
        self.is_trained = False
        
    def train(self, job_texts, job_metadata, labels):
        """
        Train the entire fraud detection pipeline.
        
        Args:
            job_texts (list): List of job posting text descriptions
            job_metadata (pd.DataFrame or np.ndarray): Metadata features
            labels (np.ndarray): Ground truth labels (0 for legitimate, 1 for fraud)
        """
        print("Training NLP stream...")
        nlp_features = self.nlp_stream.extract_features(job_texts)
        
        print("Training outlier stream...")
        self.outlier_stream.fit(job_metadata)
        outlier_features = self.outlier_stream.predict(job_metadata).reshape(-1, 1)
        
        print("Training fusion layer...")
        self.fusion_layer.fit(nlp_features, outlier_features, labels)
        
        self.is_trained = True
        print("Training complete!")
        
    def predict(self, job_texts, job_metadata):
        """
        Predict fraud for new job postings.
        
        Args:
            job_texts (list): List of job posting text descriptions
            job_metadata (pd.DataFrame or np.ndarray): Metadata features
            
        Returns:
            dict: Predictions with probabilities and explanations
        """
        if not self.is_trained:
            raise ValueError("System must be trained before prediction. Call train() first.")
            
        # Extract features from both streams
        nlp_features = self.nlp_stream.extract_features(job_texts)
        outlier_features = self.outlier_stream.predict(job_metadata).reshape(-1, 1)
        
        # Get predictions and probabilities
        predictions = self.fusion_layer.predict(nlp_features, outlier_features)
        probabilities = self.fusion_layer.predict_proba(nlp_features, outlier_features)
        
        # Get SHAP explanations
        explanations = self.fusion_layer.explain(nlp_features, outlier_features)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'explanations': explanations,
            'nlp_features': nlp_features,
            'outlier_features': outlier_features
        }
    
    def evaluate(self, job_texts, job_metadata, labels):
        """
        Evaluate the system on test data.
        
        Args:
            job_texts (list): List of job posting text descriptions
            job_metadata (pd.DataFrame or np.ndarray): Metadata features
            labels (np.ndarray): Ground truth labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("System must be trained before evaluation. Call train() first.")
            
        results = self.predict(job_texts, job_metadata)
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'roc_auc': roc_auc_score(labels, probabilities)
        }
        
        return metrics


def main():
    """
    Example usage of the Guardian Recruit Fraud Detection system.
    """
    print("=" * 60)
    print("Guardian Recruit Fraud Detection System")
    print("=" * 60)
    
    # Example: Create synthetic data for demonstration
    print("\nGenerating example data...")
    
    # Synthetic job texts
    job_texts = [
        "Software Engineer position at reputable company. Competitive salary and benefits.",
        "URGENT! Make $10000 per week from home! No experience needed! Apply now!",
        "Marketing Manager role. 5+ years experience required. Submit resume and cover letter.",
        "Work from home! Easy money! Send $500 registration fee to get started!"
    ]
    
    # Synthetic metadata (e.g., salary, company age, etc.)
    job_metadata = pd.DataFrame({
        'salary_min': [80000, 10000, 70000, 5000],
        'salary_max': [120000, 999999, 90000, 999999],
        'company_age': [10, 0, 15, 0],
        'num_requirements': [5, 0, 7, 0],
        'has_contact_info': [1, 0, 1, 0]
    })
    
    # Labels (0 = legitimate, 1 = fraud)
    labels = np.array([0, 1, 0, 1])
    
    # Initialize and train the system
    print("\nInitializing fraud detection system...")
    detector = GuardianRecruitFraudDetector(model_name='bert-base-uncased')
    
    print("\nTraining the system...")
    print("Note: In a real scenario, you would use a larger dataset.")
    print("This is just a demonstration with minimal data.")
    
    # For demonstration, we'll just show the structure
    # In practice, you would call: detector.train(job_texts, job_metadata, labels)
    
    print("\nSystem components initialized:")
    print(f"  - NLP Stream: {detector.nlp_stream.__class__.__name__}")
    print(f"  - Outlier Stream: {detector.outlier_stream.__class__.__name__}")
    print(f"  - Fusion Layer: {detector.fusion_layer.__class__.__name__}")
    
    print("\nTo train the system, call: detector.train(job_texts, job_metadata, labels)")
    print("To make predictions, call: detector.predict(job_texts, job_metadata)")
    print("To evaluate, call: detector.evaluate(job_texts, job_metadata, labels)")
    
    print("\n" + "=" * 60)
    print("System ready for training and deployment!")
    print("=" * 60)


if __name__ == "__main__":
    main()
