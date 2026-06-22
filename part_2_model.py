# COMP 472 - Mini Project 2 (Summer 2026)
# Spam Detection AI
#
# Part 2:
#   1. Train the machine learning model
#   2. Evaluate the model
#   3. Display accuracy and confusion matrix
#   4. Show confidence score for predictions
#   5. Create spam/ham bar chart
#   6. Run interactive prediction loop

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preparation import prepare_data, load_and_clean_data


def train_model(X_train, y_train):
    # Multinomial Naive Bayes works well for text classification problems.
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Model training completed.\n")
    return model


def evaluate_model(model, X_test, y_test):
    # The model predicts labels for the test data.
    y_pred = model.predict(X_test)

    # Accuracy tells us how many predictions were correct.
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Confusion matrix shows correct and wrong predictions (includes false positives and false negatives).
    labels = ["spam", "ham"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Use Pandas DataFrame for better matrix formatting
    cm_df = pd.DataFrame(
        cm, 
        index = ["Actual Spam", "Actual Ham"], 
        columns = ["Predicted Spam", "Predicted Ham"]
    )

    print("\nConfusion Matrix:")
    print(cm_df)
    print()

    return accuracy, cm


def show_label_chart(df):
    
    counts = df["label"].value_counts()

    # Display exact count numbers
    axes = counts.plot(kind="bar")
    axes.bar_label(axes.containers[0])

    plt.title("Number of Spam and Ham Messages")
    plt.xlabel("Label")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    plt.show()


def predict_message(model, vectorizer, message):
    # Convert the user's message into TF-IDF numbers using the same vectorizer.
    message_vector = vectorizer.transform([message])

    # Predict spam or ham.
    prediction = model.predict(message_vector)[0]

    # predict_proba gives confidence for each label.
    probabilities = model.predict_proba(message_vector)[0]
    confidence = max(probabilities) * 100

    print("Prediction:", prediction.upper())
    print("Confidence:", round(confidence, 2), "%\n")


def interactive_loop(model, vectorizer):
    print("Type 'quit' to exit.\n")

    while True:
        message = input("Enter message: ")

        if message.lower() == "quit":
            print("Goodbye!")
            break

        predict_message(model, vectorizer, message)


if __name__ == "__main__":
    main()