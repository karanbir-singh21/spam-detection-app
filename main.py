# COMP 472 - Mini Project 2 (Summer 2026)
# Spam Detection AI 
# Main Program Execution + Conversation Loop

# import data loading, feature extraction, feature and label splitting from part 1
from data_preparation import prepare_data

# import model training, evaluation, accuracy, visualization, conversation loop from part 2
from part_2_model import train_model, evaluate_model, show_label_chart, interactive_loop


def main():
    print("Welcome to Spam Detection AI\n")
    csv_path = "spam.csv"

    # load and clean up data, extract features, split clearly into train and test sets
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, vectorizer, df = prepare_data(csv_path)

    # visualize spam and ham data with a bar chart
    print("Visualizing data...")
    show_label_chart(df)

    # training model
    print("Training model...")
    model = train_model(X_train, y_train)

    # evaluate model, show accuracy, confusion matrix with labels
    evaluate_model(model, X_test, y_test)

    # user conversation loop for testing new messages
    interactive_loop(model, vectorizer)


if __name__ == "__main__":
    main()