"""
Outlier Stream Module for Guardian Recruit Fraud Detection
Uses Isolation Forest for metadata anomaly detection
"""

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class OutlierStream:
    """
    Outlier detection stream for analyzing job posting metadata using Isolation Forest
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the Outlier Stream with Isolation Forest
        
        Args:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False
        
    def extract_metadata_features(self, metadata):
        """
        Extract numerical features from job posting metadata
        
        Args:
            metadata (dict): Job posting metadata
            
        Returns:
            np.ndarray: Extracted features
        """
        features = []
        
        # Example metadata features
        features.append(metadata.get('salary_min', 0))
        features.append(metadata.get('salary_max', 0))
        features.append(metadata.get('experience_required', 0))
        features.append(metadata.get('num_requirements', 0))
        features.append(metadata.get('company_age', 0))
        features.append(metadata.get('has_company_logo', 0))
        features.append(metadata.get('has_contact_info', 0))
        features.append(metadata.get('posting_length', 0))
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, metadata_list):
        """
        Fit the Isolation Forest model on metadata
        
        Args:
            metadata_list (list): List of metadata dictionaries
        """
        features_list = [self.extract_metadata_features(m).flatten() 
                        for m in metadata_list]
        X = np.array(features_list)
        
        self.model.fit(X)
        self.is_fitted = True
        
    def predict_outlier_score(self, metadata):
        """
        Predict outlier score for given metadata
        
        Args:
            metadata (dict): Job posting metadata
            
        Returns:
            dict: Outlier analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        features = self.extract_metadata_features(metadata)
        
        # Get anomaly score and prediction
        anomaly_score = self.model.decision_function(features)[0]
        is_outlier = self.model.predict(features)[0]
        
        return {
            'anomaly_score': anomaly_score,
            'is_outlier': bool(is_outlier == -1),
            'features': features.flatten()
        }
    
    def analyze_metadata(self, metadata):
        """
        Analyze metadata for anomalies
        
        Args:
            metadata (dict): Job posting metadata
            
        Returns:
            dict: Analysis results
        """
        features = self.extract_metadata_features(metadata)
        
        if self.is_fitted:
            return self.predict_outlier_score(metadata)
        else:
            return {
                'features': features.flatten(),
                'message': 'Model not fitted yet'
            }
