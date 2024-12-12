import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train_model(data):
    """
    Train a logistic regression model with the cleaned data.
    
    Args:
        data (pd.DataFrame): Cleaned data with 'content' (text) and 'label' (target) columns.
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline.
    """
    # Prepare features and target variable
    X = data['content']
    y = data['label']
    
    # Create a TF-IDF vectorizer and a logistic regression model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    pipeline.fit(X, y)
    
    return pipeline

def main():
    # Load cleaned data
    try:
        cleaned_data = pd.read_csv('data/processed/cleaned_data.csv', sep=';')
        print(f"Cleaned data loaded successfully. Shape: {cleaned_data.shape}")
        
        # Train the model
        model = train_model(cleaned_data)
        print("Model trained successfully.")
        
        # Evaluate the model (optional)
        X_train, X_val, y_train, y_val = train_test_split(
            cleaned_data['content'], cleaned_data['label'], test_size=0.2, random_state=42
        )
        y_pred_val = model.predict(X_val)
        print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Classification Report:\n{classification_report(y_val, y_pred_val)}")
        
        # Save the model to disk
        joblib.dump(model, 'models/model_pipeline.pkl')
        print("Model saved to 'models/model_pipeline.pkl'")
    
    except Exception as e:
        print(f"Error training model: {e}")

if __name__ == "__main__":
    main()
