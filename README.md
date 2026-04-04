# Fake-News-Detection-Tracker
MLflow Fake News Detection

A Machine Learning project that detects fake news using NLP techniques and tracks experiments using MLflow.

 Overview

This project builds a fake news classification system using text data and compares multiple machine learning models.
It uses MLflow to log experiments, track performance metrics, and manage trained models.

## Features
Fake news classification using NLP
Multiple model comparison (Naive Bayes, Logistic Regression)
Experiment tracking with MLflow
Logging of parameters, metrics, and models
Model saving and versioning
Tech Stack
Python
Scikit-learn
MLflow
Pandas
NLP (TF-IDF Vectorizer)
Git & GitHub
## Project Structure
mlflow-fake-news-detector/
│── fake_news_mlflow.py
│── README.md
 How to Run
1️ Install Dependencies
pip install mlflow scikit-learn pandas
2️ Run the Project
python fake_news_mlflow.py
3️ Start MLflow UI
mlflow ui
4️ Open in Browser
http://127.0.0.1:5000
## results
Tracks accuracy of different models
Compares multiple experiment runs
Stores trained models for reuse
## Example Use Case

Input:

"Aliens landed in India yesterday"

Output:

Fake News
## Future Improvements
Use real-world dataset (Kaggle)
Deploy using FastAPI
Add frontend interface
Improve model accuracy with advanced NLP
# Author

Aditi Verma

## Acknowledgment

This project demonstrates practical usage of MLflow for experiment tracking and model management in machine learning workflows.
