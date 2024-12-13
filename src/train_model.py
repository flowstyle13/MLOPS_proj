import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the cleaned data from a CSV file."""
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def train_and_evaluate_model(train_df):
    """
    Train a machine learning model using Logistic Regression and evaluate its performance.
    Args:
        train_df (pd.DataFrame): The training dataset.
    Returns:
        model_pipeline: The trained model pipeline.
    """
    # Prepare features and labels for training
    X_train = train_df['content']
    y_train = train_df['label']

    # Step 4: Vectorization using TF-IDF and Model Pipeline
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

    # Create a pipeline with TF-IDF and Logistic Regression
    model_pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Step 5: Model Training
    model_pipeline.fit(X_train, y_train)

    # Step 6: Evaluation on the Training Set
    y_pred_train = model_pipeline.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred_train)
    report = classification_report(y_train, y_pred_train)
    conf_matrix = confusion_matrix(y_train, y_pred_train)

    print("Training Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)

    return model_pipeline, accuracy, report, conf_matrix

def main():
    # Load cleaned data
    train_data = load_data('data/processed/cleaned_data.csv')

    if train_data is None:
        print("Error loading the data. Exiting.")
        return

    # Start MLflow run
    with mlflow.start_run():

        # Log hyperparameters (if any)
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("stop_words", "english")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Train and evaluate model using only the training data
        model, accuracy, report, conf_matrix = train_and_evaluate_model(train_data)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the classification report and confusion matrix as artifacts
        mlflow.log_artifact("classification_report.txt", "classification_report.txt")
        mlflow.log_artifact("confusion_matrix.txt", "confusion_matrix.txt")

        # Save the classification report and confusion matrix to files
        with open("classification_report.txt", "w") as f:
            f.write(report)

        with open("confusion_matrix.txt", "w") as f:
            f.write(str(conf_matrix))

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
