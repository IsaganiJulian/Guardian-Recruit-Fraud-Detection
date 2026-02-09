# Guardian Recruit Fraud Detection - Source Code

This directory contains the core modules of the fraud detection system.

## Modules

### nlp_stream.py
BERT/RoBERTa text analysis module for processing job posting descriptions.
- Uses transformer models to extract semantic features from text
- Supports custom pre-trained models
- Returns embeddings and fraud probabilities

### outlier_stream.py
Isolation Forest module for metadata anomaly detection.
- Analyzes job posting metadata (salary, company info, etc.)
- Identifies outliers using unsupervised learning
- Provides anomaly scores and feature importance

### fusion_layer.py
XGBoost meta-classifier with SHAP explainability.
- Combines predictions from NLP and outlier streams
- Uses gradient boosting for final classification
- Provides SHAP values for model interpretability

### main.py
System integration and example usage.
- Demonstrates how to use all components together
- Provides training, prediction, and evaluation pipelines
- Can be run as a standalone script

## Installation

Install dependencies from the root directory:

```bash
pip install -r ../requirements.txt
```

## Usage

```python
from main import GuardianRecruitFraudDetector

# Initialize the detector
detector = GuardianRecruitFraudDetector()

# Train on your data
detector.train(job_texts, job_metadata, labels)

# Make predictions
results = detector.predict(new_job_texts, new_job_metadata)

# Evaluate performance
metrics = detector.evaluate(test_texts, test_metadata, test_labels)
```

## Running the Example

```bash
python main.py
```

Note: The example in main.py requires all dependencies to be installed first.
