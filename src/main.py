"""
Main Module for Guardian Recruit Fraud Detection System
System integration and orchestration
"""

import numpy as np
from nlp_stream import NLPStream
from outlier_stream import OutlierStream
from fusion_layer import FusionLayer


class GuardianRecruitFraudDetector:
    """
    Main fraud detection system that integrates all components
    """
    
    def __init__(self):
        """Initialize the fraud detection system"""
        self.nlp_stream = NLPStream()
        self.outlier_stream = OutlierStream()
        self.fusion_layer = FusionLayer()
        self.is_trained = False
        
    def train(self, job_postings, labels):
        """
        Train the fraud detection system
        
        Args:
            job_postings (list): List of job posting dictionaries with 'text' and 'metadata'
            labels (np.ndarray): Binary labels (0: legitimate, 1: fraud)
        """
        print("Training Guardian Recruit Fraud Detection System...")
        
        # Extract NLP features
        print("Extracting NLP features...")
        nlp_features = []
        for posting in job_postings:
            text = posting.get('text', '')
            features = self.nlp_stream.analyze_text(text)['features']
            nlp_features.append(features)
        
        # Train outlier detector and extract features
        print("Training outlier detector...")
        metadata_list = [posting.get('metadata', {}) for posting in job_postings]
        self.outlier_stream.fit(metadata_list)
        
        outlier_features = []
        for metadata in metadata_list:
            features = self.outlier_stream.analyze_metadata(metadata)['features']
            outlier_features.append(features)
        
        # Train fusion layer
        print("Training fusion layer...")
        self.fusion_layer.fit(nlp_features, outlier_features, labels)
        
        self.is_trained = True
        print("Training complete!")
        
    def predict(self, job_posting):
        """
        Predict if a job posting is fraudulent
        
        Args:
            job_posting (dict): Job posting with 'text' and 'metadata'
            
        Returns:
            dict: Prediction results with explanation
        """
        if not self.is_trained:
            raise ValueError("System must be trained before making predictions")
        
        # Extract features
        text = job_posting.get('text', '')
        metadata = job_posting.get('metadata', {})
        
        nlp_result = self.nlp_stream.analyze_text(text)
        outlier_result = self.outlier_stream.analyze_metadata(metadata)
        
        # Get prediction with explanation
        result = self.fusion_layer.explain_prediction(
            nlp_result['features'],
            outlier_result['features']
        )
        
        return {
            'is_fraud': bool(result['prediction'] == 1),
            'fraud_probability': result['fraud_probability'],
            'legitimate_probability': result['legitimate_probability'],
            'explanation': {
                'top_feature_indices': result['top_feature_indices'],
                'top_feature_importance': result['top_feature_importance']
            }
        }
    
    def analyze_batch(self, job_postings):
        """
        Analyze a batch of job postings
        
        Args:
            job_postings (list): List of job posting dictionaries
            
        Returns:
            list: List of prediction results
        """
        results = []
        for posting in job_postings:
            result = self.predict(posting)
            results.append(result)
        return results


def main():
    """
    Main entry point for the fraud detection system
    """
    print("Guardian Recruit Fraud Detection System")
    print("=" * 50)
    
    # Initialize the system
    detector = GuardianRecruitFraudDetector()
    
    # Example usage
    print("\nSystem initialized successfully!")
    print("\nTo use the system:")
    print("1. Prepare training data with job postings and labels")
    print("2. Call detector.train(job_postings, labels)")
    print("3. Use detector.predict(job_posting) for predictions")
    
    # Example job posting structure
    example_posting = {
        'text': 'We are hiring! Great opportunity with competitive salary...',
        'metadata': {
            'salary_min': 50000,
            'salary_max': 80000,
            'experience_required': 3,
            'num_requirements': 5,
            'company_age': 10,
            'has_company_logo': 1,
            'has_contact_info': 1,
            'posting_length': 500
        }
    }
    
    print("\nExample job posting structure:")
    print(example_posting)


if __name__ == "__main__":
    main()
