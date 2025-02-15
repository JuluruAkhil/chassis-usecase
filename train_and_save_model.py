# train_and_save_model.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE

import joblib  # for saving the model
from datetime import datetime


def calculate_months_to_expiry(expiry, current_date=None):
    """
    Converts an expiry string in the format 'MM/YY' to the number of months
    from 'current_date' until the card expires.
    """
    if current_date is None:
        current_date = datetime.now()
    exp_month, exp_year = map(int, expiry.split("/"))
    expiry_date = datetime(year=2000 + exp_year, month=exp_month, day=1)
    return max(
        0,
        (expiry_date.year - current_date.year) * 12
        + (expiry_date.month - current_date.month),
    )


def train_and_save_model(csv_path="data2.csv", model_path="my_fraud_model.pkl"):
    """
    Loads the dataset, trains a Random Forest model, and saves the model to a file.
    """

    # 1. Load the dataset
    data = pd.read_csv(csv_path)

    # 2. Convert 'Expiry' to a numeric feature
    data["Months_to_expiry"] = data["Expiry"].apply(calculate_months_to_expiry)

    # 3. One-hot encode the 'Profession' column
    data = pd.get_dummies(data, columns=["Profession"], drop_first=True)

    # 4. Drop irrelevant columns
    #    (Ensure 'Credit_card_number' and 'Expiry' are in your dataset before dropping)
    data.drop(columns=["Credit_card_number", "Expiry"], inplace=True)

    # 5. Split data into features and target
    X = data.drop(columns=["Fraud"])
    y = data["Fraud"]

    # 6. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 7. Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 8. Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # 9. Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # If you want to view feature importances during training:
    # importances = pd.Series(model.feature_importances_, index=X.columns)
    # importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
    # plt.show()

    # 10. Save the model
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}.")


if __name__ == "__main__":
    # Run training when this file is executed
    train_and_save_model(csv_path="data2.csv", model_path="my_fraud_model.pkl")
