# load_and_use_model.py

import joblib
import pandas as pd


def load_model(model_path="my_fraud_model.pkl"):
    """
    Loads the saved model from file.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}.")
    return model


def make_predictions(model, new_data):
    """
    Uses the loaded model to predict fraud probabilities or classes.
    new_data should be a pandas DataFrame or 2D list/array with the same columns
    (and column order) used during training.
    """
    # If new_data is a raw list/array, ensure it matches the feature order from training:
    # new_data = pd.DataFrame(new_data, columns=[...])  # list your training features here

    y_pred = model.predict(new_data)
    y_prob = model.predict_proba(new_data)[:, 1]  # Probability of class "1" (Fraud)
    return y_pred, y_prob


if __name__ == "__main__":
    # Example usage:

    # 1. Load the trained model
    clf = load_model("my_fraud_model.pkl")

    # 2. Suppose you have some new samples to predict on:
    #    Make sure the structure & columns match what the model expects.
    #    For demonstration, we create an empty DataFrame with the same columns
    #    you used in training (except 'Fraud'):

    feature_columns = [
        "Income",
        "Security_code",
        "Months_to_expiry",
        "Profession_ENGINEER",
        "Profession_LAWYER",
    ]
    # Example data point (replace with real values).
    # This must match the exact columns and order used during training.
    new_data_df = new_data_df = pd.DataFrame(
        {
            "Income": [75000, 120000],
            "Security_code": [1234, 5678],
            "Months_to_expiry": [12, 8],
            "Profession_ENGINEER": [1, 0],
            "Profession_LAWYER": [0, 1],
        }
    )

    # 3. Make predictions
    pred_class, pred_prob = make_predictions(clf, new_data_df)
    print("Predicted Class:", pred_class)
    print("Predicted Probability (of Fraud):", pred_prob)
