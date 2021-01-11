# Disaster-Tweets


## Overview
Predict whether a tweet is about a real disaster or not. The dataset is taken from the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview) Kaggle competition. The best model ranks #134 out of 1350 teams in the leaderboard (Top 10%) as of **Jan 09, 2021**.
Best model is a stacking of:
- Modified base BERT
- LSTM with GloVe embeddings
- Naive Bayes classifier with Tf-idf features
- Custom features
- Stacking classifier is a Ridge (L2) regression model with a high regularization coefficient

Final leaderboard F1 score: **0.84094**


## Files
**input**: Hosts the dataset, submission template and extra dataset to expand the train set

**output**: Submission csv

**main.py**: Script with the main code (optimization of individual classifiers was done separately)

**utils.py**: Script with helper functions
