# Spam Detection AI — COMP 472 Mini Project 2

Classifies SMS messages as **Spam** or **Ham** using TF-IDF feature extraction and a Naive Bayes classifier.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

Make sure `spam.csv` is in the same directory, then run:

```bash
python spam_detector.py
```

The program will train the model, display accuracy and a confusion matrix, show a spam/ham bar chart, then enter an interactive loop:

```
Welcome to Spam Detection AI
Training model...
Accuracy: 96.8%

Enter message: Congratulations! You won $5000.
Prediction: SPAM
Confidence: 99.1%

Enter message: quit
Goodbye!
```

## Dataset

Uses the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (`spam.csv`).

## Output

- Accuracy score and confusion matrix (printed + chart)
- Bar chart: spam vs ham message counts
- Per-message prediction label and confidence score
