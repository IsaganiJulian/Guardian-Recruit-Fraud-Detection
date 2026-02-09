"""
Outlier Stream Module - Isolation Forest for Metadata Analysis
Handles anomaly detection in job posting metadata using Isolation Forest.
"""

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class OutlierStream:
    """
    Metadata analysis stream using Isolation Forest for detecting anomalous job postings.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the outlier detection stream.
        
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
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_metadata_features(self, data):
        """
        Extract metadata features from job posting data.
        
        Args:
            data (pd.DataFrame): DataFrame containing job posting metadata
            
        Returns:
            np.ndarray: Extracted metadata features
        """
        features = []
        
        # Example metadata features that could be extracted:
        # - Salary range (min, max, spread)
        # - Company age
        # - Number of requirements
        # - Application deadline urgency
        # - Contact information completeness
        
        if isinstance(data, pd.DataFrame):
            feature_columns = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
            features = data[feature_columns].values
        else:
            features = np.array(data)
            
        return features
    
    def fit(self, data):
        """
        Fit the Isolation Forest model on metadata features.
        
        Args:
            data (pd.DataFrame or np.ndarray): Training data
        """
        features = self.extract_metadata_features(data)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        self.is_fitted = True
        
    def predict(self, data):
        """
        Predict outlier scores for given metadata.
        
        Args:
            data (pd.DataFrame or np.ndarray): Data to analyze
            
        Returns:
            np.ndarray: Outlier scores (higher is more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
            
        features = self.extract_metadata_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly scores (negative scores are more anomalous)
        scores = self.model.score_samples(features_scaled)
        
        # Convert to positive scores where higher = more anomalous
        anomaly_scores = -scores
        
        return anomaly_scores
    
    def predict_labels(self, data):
        """
        Predict binary labels (outlier or not) for given metadata.
        
        Args:
            data (pd.DataFrame or np.ndarray): Data to analyze
            
        Returns:
            np.ndarray: Binary labels (-1 for outlier, 1 for inlier)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
            
        features = self.extract_metadata_features(data)
        features_scaled = self.scaler.transform(features)
        
        return self.model.predict(features_scaled)
    
    def get_feature_importance(self, data, feature_names=None):
        """
        Get feature importance based on contribution to anomaly detection.
        
        Args:
            data (pd.DataFrame or np.ndarray): Data to analyze
            feature_names (list): Names of features
            
        Returns:
            dict: Feature importance scores
        """
        features = self.extract_metadata_features(data)
        
        if feature_names is None:
            if isinstance(data, pd.DataFrame):
                feature_names = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
            else:
                feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Calculate variance contribution as a proxy for importance
        importance = np.var(features, axis=0)
        importance = importance / np.sum(importance)
        
        return dict(zip(feature_names, importance))
