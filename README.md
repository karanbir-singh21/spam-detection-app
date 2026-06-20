# COMP 472 - Mini Project 2: Spam Detection AI

An AI system that classifies SMS/email messages as **spam** or **ham** (not spam).

## Requirements
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data preparation (`data_preparation.py`)
Handles loading, cleaning, feature extraction and the train/test split:

1. **Load + clean** the dataset (`spam.csv`) with pandas.
2. **Feature extraction** — convert text to numbers with `TfidfVectorizer`.
3. **Split** into training and testing sets with `train_test_split` (80/20).

Run it on its own to test:
```
python data_preparation.py
```

It exposes `prepare_data("spam.csv")`, which returns everything the model
training step needs:
```python
from data_preparation import prepare_data

X_train, X_test, y_train, y_test, vectorizer = prepare_data("spam.csv")
```

## Dataset
`spam.csv` — SMS Spam Collection dataset (5,572 messages: 4,825 ham / 747 spam).
