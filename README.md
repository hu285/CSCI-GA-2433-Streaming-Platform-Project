# Streaming Platform Database Project

This repository contains the end-to-end application developed for Part 4 of my project.

## Technologies
- Python
- Tkinter
- SQLite
- SQLAlchemy ORM
- scikit-learn
- TF-IDF
- Logistic Regression

## Main Files
- streaming_platform_app.py – end-to-end user application
- sentiment_module.py – sentiment analysis and retraining pipeline
- streaming_platform.db – SQLite operational database
- sentiment_model.joblib – trained sentiment model
- sentiment_model_metadata.json – model and source-data metadata

## Dataset
The sentiment model uses an IMDb dataset of 50K movie reviews. Since the dataset 
is too large to be uploaded here, it can be downloaded separately. The file titled
"IMDB Dataset.csv" can be placed in the same directory as the Python scripts if 
model retraining is required.

## Run
After installing the required Python packages, the following script can be run:

python streaming_platform_app.py
