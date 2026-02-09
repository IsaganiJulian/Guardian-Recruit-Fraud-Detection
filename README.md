# Guardian Recruit Fraud Detection

This AI-powered system helps ensure job seekers' safety and mitigate potential harm from fraudulent job postings.

## Overview

Guardian Recruit uses advanced machine learning techniques to detect fraudulent job postings by analyzing both textual content and metadata. The system combines:

- **NLP Stream**: BERT/RoBERTa transformer models for semantic text analysis
- **Outlier Stream**: Isolation Forest for metadata anomaly detection
- **Fusion Layer**: XGBoost meta-classifier with SHAP explainability

## Project Structure

```
.
├── data/          # Training and test data
├── notebooks/     # Jupyter notebooks for analysis
├── models/        # Trained model artifacts
├── src/           # Source code modules
│   ├── nlp_stream.py      # Text analysis module
│   ├── outlier_stream.py  # Metadata analysis module
│   ├── fusion_layer.py    # Meta-classifier module
│   └── main.py            # System integration
└── requirements.txt       # Python dependencies
```

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

## Quick Start

```python
from src.main import GuardianRecruitFraudDetector
import pandas as pd

# Initialize the fraud detection system
detector = GuardianRecruitFraudDetector()

# Prepare your data
job_texts = ["Job description 1", "Job description 2", ...]
job_metadata = pd.DataFrame({...})
labels = [0, 1, 0, ...]  # 0 = legitimate, 1 = fraud

# Train the system
detector.train(job_texts, job_metadata, labels)

# Make predictions
results = detector.predict(new_job_texts, new_job_metadata)
print(f"Fraud probability: {results['probabilities']}")
```

## Features

- **Multi-Modal Analysis**: Combines text and metadata for robust detection
- **State-of-the-Art NLP**: Uses pre-trained transformer models (BERT/RoBERTa)
- **Anomaly Detection**: Identifies unusual patterns in job posting metadata
- **Explainable AI**: SHAP values provide interpretable predictions
- **Flexible Architecture**: Easy to extend and customize

## Requirements

- Python 3.7+
- transformers
- scikit-learn
- xgboost
- shap
- torch
- numpy
- pandas

See `requirements.txt` for complete list.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
