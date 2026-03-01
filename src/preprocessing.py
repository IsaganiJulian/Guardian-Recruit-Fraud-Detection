import pandas as pd
import numpy as np

class GuardianCleaner:
    def __init__(self):
        self.binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
        self.text_cols = ['description', 'requirements', 'benefits', 'company_profile']

    def _clean_types(self, df):
        # Standardize obvious null-like values
        df.replace(['', ' ', 'none', 'None', 'NaN', 'unknown', 'Unknown'], np.nan, inplace=True)

        # Robust boolean mapping for t/f, true/false, 1/0
        bool_map = {
            't': 1, 'true': 1, '1': 1,
            'f': 0, 'false': 0, '0': 0,
        }

        cols_to_binary = self.binary_cols + ['fraudulent', 'in_balanced_dataset']

        for col in cols_to_binary:
            if col in df.columns:
                s = df[col].astype(str).str.strip().str.lower()
                s = s.map(bool_map)                 # unknown tokens -> NaN
                df[col] = s.fillna(0).astype(int)  # default unknowns to 0

        return df


    def _feature_engineering(self, df):
        """Internal helper for extraction and NLP safety."""
        # Location Extraction
        if 'location' in df.columns:
            location_series = df['location'].fillna('Unknown').astype(str)
            df['country'] = location_series.str.split(',').str[0].str.strip().replace('', 'Unknown')
        
        # NLP Placeholder Handling (Prevents crashes)
        for col in self.text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unspecified')
        
        # New "Robust" Feature: Text Length
        if 'description' in df.columns:
            df['desc_len'] = df['description'].astype(str).str.len()
            
        return df

    def transform(self, df, is_training=True):
        """The main pipeline execution."""
        data = df.copy()
        
        # 1. Nulls & Types
        data = self._clean_types(data)
        
        # 2. Features & NLP Safety
        data = self._feature_engineering(data)
        
        # 3. Deduplication (Only for training)
        if is_training:
            data = data.drop_duplicates()
            
        return data
