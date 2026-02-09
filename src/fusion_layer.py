"""
Fusion Layer Module - XGBoost Meta-Classifier with SHAP Explainability
Combines outputs from NLP and outlier streams for final fraud detection.
"""

import xgboost as xgb
import shap
import numpy as np
import pandas as pd


class FusionLayer:
    """
    Meta-classifier that fuses predictions from multiple streams using XGBoost.
    Includes SHAP for explainable AI (XAI).
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        """
        Initialize the fusion layer with XGBoost classifier.
        
        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate for XGBoost
            max_depth (int): Maximum tree depth
            random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        self.explainer = None
        self.is_fitted = False
        
    def fuse_features(self, nlp_features, outlier_features):
        """
        Combine features from NLP and outlier streams.
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            
        Returns:
            np.ndarray: Combined feature matrix
        """
        # Ensure both inputs are 2D arrays
        if len(nlp_features.shape) == 1:
            nlp_features = nlp_features.reshape(-1, 1)
        if len(outlier_features.shape) == 1:
            outlier_features = outlier_features.reshape(-1, 1)
            
        # Concatenate features horizontally
        fused_features = np.hstack([nlp_features, outlier_features])
        
        return fused_features
    
    def fit(self, nlp_features, outlier_features, labels, validation_split=0.2):
        """
        Train the meta-classifier on fused features.
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            labels (np.ndarray): Ground truth labels (0 or 1)
            validation_split (float): Proportion of data to use for validation
        """
        X = self.fuse_features(nlp_features, outlier_features)
        
        # Split into train and validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.is_fitted = True
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
    def predict(self, nlp_features, outlier_features):
        """
        Predict fraud labels for given features.
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
            
        X = self.fuse_features(nlp_features, outlier_features)
        return self.model.predict(X)
    
    def predict_proba(self, nlp_features, outlier_features):
        """
        Predict fraud probabilities for given features.
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            
        Returns:
            np.ndarray: Fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
            
        X = self.fuse_features(nlp_features, outlier_features)
        return self.model.predict_proba(X)[:, 1]
    
    def explain(self, nlp_features, outlier_features):
        """
        Generate SHAP explanations for predictions.
        
        Args:
            nlp_features (np.ndarray): Features from NLP stream
            outlier_features (np.ndarray): Features from outlier stream
            
        Returns:
            shap.Explanation: SHAP values explaining the predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation. Call fit() first.")
            
        X = self.fuse_features(nlp_features, outlier_features)
        shap_values = self.explainer(X)
        
        return shap_values
    
    def get_feature_importance(self):
        """
        Get feature importance from the XGBoost model.
        
        Returns:
            dict: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call fit() first.")
            
        importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
