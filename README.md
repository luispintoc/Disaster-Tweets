# Disaster-Tweets


## Overview
Predict whether a tweet is about a real disaster. The dataset is taken from the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview) Kaggle competition. The best model ranks #130 in the leaderboard (Top 10%) as of **Jan 09, 2021**. 
Best model is a stacking of:
- Modified BERT
- Bidirectional LSTM with GloVe embeddings
- Logistic Regression using Tf-idf features
- Custom features
- Stacking classifier is a SVM


## Files
**input**: Hosts the dataset, submission template and extra dataset to expand the train set
**output**: Submission csv
**main.py**: Script with the main code (optimization of individual classifiers was done separately)
**utils.py**: Script with helper functions
