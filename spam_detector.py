"""
Spam Detection AI - COMP 472 Mini Project 2
Classifies SMS/email messages as Spam or Ham (not spam) using TF-IDF
feature extraction and a Naive Bayes classifier.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATASET_PATH = "spam.csv"


def load_dataset(path):
    """Load the SMS Spam Collection dataset into a (label, message) DataFrame."""
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        print(f"Error: could not find dataset file '{path}'.")
        sys.exit(1)

    # The provided CSV ships with a typo'd header ("lable") and trailing
    # blank columns, so normalize it to the expected label/message format.
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    return df


def plot_class_distribution(df):
    """Bar chart of spam vs ham message counts."""
    counts = df["label"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color=["#d9534f", "#5cb85c"])
    plt.title("Spam vs Ham Message Counts")
    plt.xlabel("Label")
    plt.ylabel("Number of Messages")
    for i, value in enumerate(counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("Saved class distribution chart to class_distribution.png")
    plt.close()


def train_model(df):
    """Vectorize messages with TF-IDF and train a Naive Bayes classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.05)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])

    return model, vectorizer, accuracy, cm


def print_evaluation(accuracy, cm):
    print(f"Accuracy: {accuracy * 100:.1f}%")
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Spam   Ham")
    print(f"Actual Spam   {cm[0][0]:<6} {cm[0][1]:<6}")
    print(f"Actual Ham    {cm[1][0]:<6} {cm[1][1]:<6}")


def plot_confusion_matrix(cm):
    """Heatmap of the confusion matrix saved to confusion_matrix.png."""
    cm_array = np.array(cm)
    labels = ["Spam", "Ham"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix heatmap to confusion_matrix.png")
    plt.close()


def predict_message(message, model, vectorizer):
    """Predict the label of a message and return (label, confidence)."""
    vec = vectorizer.transform([message])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = probs.argmax()
    return classes[best_idx], probs[best_idx]


def interactive_loop(model, vectorizer):
    print("Type 'quit' to exit.")
    while True:
        message = input("Enter message:\n")
        if message.strip().lower() == "quit":
            print("Goodbye!")
            break
        if not message.strip():
            continue
        label, confidence = predict_message(message, model, vectorizer)
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence * 100:.1f}%\n")


def main():
    print("Welcome to Spam Detection AI")

    df = load_dataset(DATASET_PATH)
    plot_class_distribution(df)

    print("Training model...")
    model, vectorizer, accuracy, cm = train_model(df)

    print_evaluation(accuracy, cm)
    plot_confusion_matrix(cm)
    interactive_loop(model, vectorizer)


if __name__ == "__main__":
    main()
