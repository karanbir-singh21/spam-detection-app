# COMP 472 - Mini Project 2 (Summer 2026)
# Spam Detection AI
#
# My part (data preparation):
#   1. load + clean the csv with pandas
#   2. turn the text into numbers with TfidfVectorizer
#   3. split into train/test with train_test_split
# Then I hand the train/test data + vectorizer to the model training part.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_and_clean_data(csv_path):
    # the spam.csv is a bit messy: header is spelled "lable", and there are
    # a few empty extra columns at the end, so I clean those up here.
    df = pd.read_csv(csv_path, encoding="latin-1")

    # only keep the first 2 columns (label + message), drop the empty ones
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    # drop empty rows and tidy up the labels
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.reset_index(drop=True)

    print("Loaded", len(df), "messages")
    print(df["label"].value_counts(), "\n")
    return df


def extract_features(df):
    # computers can't read words, so TF-IDF converts each message into a vector.
    # common words get low weight, rarer "spammy" words get higher weight.
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    print("Feature matrix shape:", X.shape, "\n")
    return X, y, vectorizer


def split_data(X, y):
    # 80% train / 20% test. stratify keeps the spam/ham ratio the same in both.
    # random_state so the split is the same every time we run it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train:", X_train.shape[0], " Test:", X_test.shape[0], "\n")
    return X_train, X_test, y_train, y_test


def prepare_data(csv_path):
    # runs all my steps and returns what the model-training part needs
    df = load_and_clean_data(csv_path)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = prepare_data("spam.csv")
    print("Data ready for model training.")
