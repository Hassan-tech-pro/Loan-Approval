# 🚀 Loan Approval Prediction System

A real-time loan approval prediction system using Machine Learning with an intuitive web interface built with Gradio.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.9+-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This project implements a loan approval prediction system that combines the power of XGBoost and HistGradientBoosting classifiers with an intuitive Gradio web interface. The system predicts loan approval status based on multiple factors including income, home ownership, loan intent, interest rates, and previous defaults.

## ✨ Features

- **Machine Learning Models**: XGBoost & HistGradientBoosting classifiers for accurate predictions
- **Smart Pre-processing**: Custom business rules for quick decision-making on low-risk applications
- **Web Interface**: Gradio-powered UI for easy interaction
- **Production-Ready**: Includes serialized model, scaler, and encoder for deployment

## 🛠️ Tech Stack

- **Python** - Programming language
- **XGBoost** - Gradient boosting framework
- **Scikit-learn** - Machine learning library
- **Gradio** - Web interface framework
- **Pandas & NumPy** - Data manipulation
- **One-Hot Encoding** - Categorical feature encoding
- **MinMax Scaling** - Feature normalization

## 📊 Features Used

The model considers the following features:
- **Annual Income** - applicant's yearly income
- **Home Ownership** - OWN, MORTGAGE, RENT, OTHER
- **Loan Intent** - Education, Medical, Venture, Home Improvement, Personal, Debt Consolidation
- **Interest Rate** - loan interest rate percentage
- **Loan Amount** - requested loan amount
- **Previous Loan Defaults** - Yes/No history of defaults

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Hassan-tech-pro/Loan-Approval.git
cd Loan-Approval
```

2. Install dependencies:
```bash
pip install xgboost scikit-learn gradio pandas numpy
```

3. Run the application:
```bash
python gradio_app.py
```

4. Open your browser and go to the local URL provided (typically `http://127.0.0.1:7860`)

## 📁 Project Structure

```
Loan-Approval/
├── data/
│   └── loan_data.csv          # Training dataset
├── posts/                      # Marketing content
├── model.pkl                  # Trained XGBoost model
├── scaler.pkl                  # Feature scaler
├── encoder.pkl                 # Categorical encoder
├── gradio_app.py              # Web application
├── loan.ipynb                 # Jupyter notebook with training code
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore file
```

## 🎨 Usage

Simply enter the applicant's information in the web interface:
1. Enter annual income
2. Select home ownership status
3. Choose loan intent
4. Indicate previous loan defaults
5. Enter interest rate
6. Enter loan amount

Click "Submit" to get instant approval/rejection prediction!

## 🔬 Model Details

- **XGBoost Classifier**: Primary gradient boosting model
- **HistGradientBoosting Classifier**: Alternative gradient boosting model
- **Feature Engineering**: One-Hot Encoding for categorical variables
- **Normalization**: MinMax Scaling for numerical features
- **Business Rules**: Smart pre-processing for quick decisions on low-risk applications

## 📈 Results

The models achieve high accuracy in predicting loan approval status, with XGBoost providing robust predictions across various scenarios.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Hassan**

- GitHub: [@Hassan-tech-pro](https://github.com/Hassan-tech-pro)

## 📌 Keywords

`machine-learning` `loan-approval` `xgboost` `gradio` `python` `data-science` `fintech` `prediction-model`

---

⭐️ If you found this project interesting, please give it a star!
