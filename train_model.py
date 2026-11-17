"""
Train the Walk/Run XGBoost model for the Streamlit application.

This script follows the notebook workflow:
1. Read walkrun.csv
2. Drop date, time and username
3. Apply the notebook's IQR median-replacement treatment
4. Split 80:20 with random_state=45
5. Train XGBoost and save model.pkl
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

DATA_PATH = "walkrun.csv"
TARGET = "activity"
FEATURES = [
    "wrist",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

data = pd.read_csv(DATA_PATH)
data = data.drop(columns=["date", "time", "username"])

# Match the IQR outlier treatment used in the original notebook.
for column in data.columns:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data.loc[(data[column] < lower) | (data[column] > upper), column] = data[column].median()

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

model = XGBClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

joblib.dump(
    {
        "model": model,
        "feature_names": FEATURES,
        "test_accuracy": float(accuracy),
        "test_f1_score": float(f1),
        "feature_medians": {name: float(X[name].median()) for name in FEATURES},
    },
    "model.pkl",
)

print(f"Test accuracy: {accuracy:.4%}")
print(f"Test F1-score: {f1:.4%}")
print("\\nClassification report:")
print(classification_report(y_test, predictions))
print("\\nSaved trained model to model.pkl")
