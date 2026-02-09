# Guardian-Recruit-Fraud-Detection

This AI-powered system helps ensure job seekers' safety and mitigate potential harm from fraudulent job postings.

## Project Structure

```
Guardian-Recruit-Fraud-Detection/
├── data/              # Data storage directory
├── notebooks/         # Jupyter notebooks for analysis
├── models/            # Trained model storage
├── src/               # Source code
│   ├── nlp_stream.py      # BERT/RoBERTa text analysis
│   ├── outlier_stream.py  # Isolation Forest for metadata anomaly detection
│   ├── fusion_layer.py    # XGBoost meta-classifier with SHAP explainability
│   └── main.py            # System integration and orchestration
└── requirements.txt   # Python dependencies
```

## Features

- **NLP Stream**: Uses BERT/RoBERTa transformer models for analyzing job posting text
- **Outlier Detection**: Implements Isolation Forest for detecting anomalies in job posting metadata
- **Fusion Layer**: XGBoost meta-classifier that combines NLP and outlier features
- **Explainability**: SHAP (SHapley Additive exPlanations) for interpretable predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IsaganiJulian/Guardian-Recruit-Fraud-Detection.git
cd Guardian-Recruit-Fraud-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.main import GuardianRecruitFraudDetector

# Initialize the detector
detector = GuardianRecruitFraudDetector()

# Prepare your training data
job_postings = [
    {
        'text': 'Job description text...',
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
    },
    # ... more postings
]
labels = [0, 1, 0, ...]  # 0: legitimate, 1: fraud

# Train the system
detector.train(job_postings, labels)

# Make predictions
new_posting = {
    'text': 'New job posting text...',
    'metadata': { ... }
}
result = detector.predict(new_posting)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']}")
```

## Components

### NLP Stream (`nlp_stream.py`)
- Utilizes pre-trained BERT/RoBERTa models
- Extracts semantic embeddings from job posting text
- Analyzes text for fraud indicators

### Outlier Stream (`outlier_stream.py`)
- Implements Isolation Forest algorithm
- Detects anomalies in job posting metadata
- Extracts numerical features from metadata

### Fusion Layer (`fusion_layer.py`)
- XGBoost meta-classifier for final predictions
- Combines NLP and outlier detection features
- SHAP explainability for transparent decisions

### Main System (`main.py`)
- Orchestrates all components
- Provides unified training and prediction interface
- Handles batch processing

## License

See LICENSE file for details.
