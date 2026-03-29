# Sentiment Analyzer

## Overview

This project is a simple sentiment analysis tool that classifies text as **positive** or **negative**. The idea came from noticing how difficult it can be to manually understand large amounts of reviews or comments.

The model is trained on a dataset of short text samples and uses basic Natural Language Processing techniques to make predictions.

---

## Problem Statement

With the increasing amount of user-generated content (reviews, comments, tweets), it becomes difficult to manually analyze opinions. This project aims to automate that process using machine learning.

---

## Approach

The workflow of the project is as follows:

1. Clean and preprocess the input text
2. Convert text into numerical features using TF-IDF
3. Train a classification model (Logistic Regression)
4. Predict sentiment based on user input

---

## Features

* Takes user input from terminal
* Classifies text as positive or negative
* Uses basic NLP preprocessing
* Fast and lightweight

---

## Tech Stack

* Python
* scikit-learn
* pandas
* nltk

---

## Project Structure

```
sentiment-analyzer/
│
├── data/
├── model/
├── train.py
├── app.py
├── utils.py
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py
```

### 3. Run the application

```
python app.py
```

---

## Example

Input:

```
This movie is really good
```

Output:

```
Positive
```

---

## Challenges Faced

* Handling noisy text (slang, punctuation, etc.)
* Choosing the right preprocessing steps
* Avoiding overfitting on small dataset

---

## What I Learned

* Basics of NLP preprocessing
* How text is converted into numerical features
* Training and saving ML models
* Importance of consistent preprocessing

---

## Future Improvements

* Add a web interface
* Support neutral sentiment
* Use larger real-world datasets
* Try advanced models (like deep learning)
