"""
Fusion Layer Module for Guardian Recruit Fraud Detection
Uses XGBoost meta-classifier with SHAP for explainability
"""

import xgboost as xgb
import shap
import numpy as np
from sklearn.model_selection import train_test_split


class FusionLayer:
    """
    Fusion layer that combines NLP and outlier detection features
    using XGBoost meta-classifier with SHAP explainability
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Initialize the Fusion Layer with XGBoost
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            objective='binary:logistic'
        )
        self.explainer = None
        self.is_fitted = False
        
    def combine_features(self, nlp_features, outlier_features):
        """
        Combine features from NLP and outlier streams
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            
        Returns:
            np.ndarray: Combined feature vector
        """
        # Convert to numpy arrays if needed
        nlp_arr = np.asarray(nlp_features)
        outlier_arr = np.asarray(outlier_features)
        
        # Flatten if needed
        nlp_flat = nlp_arr.flatten() if nlp_arr.ndim > 1 else nlp_arr
        outlier_flat = outlier_arr.flatten() if outlier_arr.ndim > 1 else outlier_arr
        
        # Concatenate features
        combined = np.concatenate([nlp_flat, outlier_flat])
        
        return combined
    
    def fit(self, nlp_features_list, outlier_features_list, labels):
        """
        Train the XGBoost meta-classifier
        
        Args:
            nlp_features_list (list): List of NLP feature arrays
            outlier_features_list (list): List of outlier feature arrays
            labels (np.ndarray): Binary labels (0: legitimate, 1: fraud)
        """
        # Combine all features
        X = np.array([
            self.combine_features(nlp_feat, outlier_feat)
            for nlp_feat, outlier_feat in zip(nlp_features_list, outlier_features_list)
        ])
        
        # Train the model
        self.model.fit(X, labels)
        self.is_fitted = True
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
    def predict(self, nlp_features, outlier_features):
        """
        Predict fraud probability for a job posting
        
        Args:
            nlp_features (np.ndarray): NLP features
            outlier_features (np.ndarray): Outlier features
            
        Returns:
            dict: Prediction results with probability and explanation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Combine features
        X = self.combine_features(nlp_features, outlier_features).reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Get SHAP values for explainability
        shap_values = self.explainer.shap_values(X)
        
        return {
            'prediction': int(prediction),
            'fraud_probability': float(probability[1]),
            'legitimate_probability': float(probability[0]),
            'shap_values': shap_values,
            'features': X.flatten()
        }
    
    def explain_prediction(self, nlp_features, outlier_features):
        """
        Generate detailed explanation for prediction using SHAP
        
        Args:
            nlp_features (np.ndarray): NLP features
            outlier_features (np.ndarray): Outlier features
            
        Returns:
            dict: Detailed explanation of the prediction
        """
        prediction_result = self.predict(nlp_features, outlier_features)
        
        # Get top contributing features
        shap_vals = prediction_result['shap_values']
        feature_importance = np.abs(shap_vals).flatten()
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        
        return {
            **prediction_result,
            'top_feature_indices': top_indices.tolist(),
            'top_feature_importance': feature_importance[top_indices].tolist()
        }
