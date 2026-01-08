import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def select_features(X, y, threshold=0.01):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    selected_features = importance_df[
        importance_df["Importance"] >= threshold
    ]["Feature"].tolist()

    joblib.dump(selected_features, "selected_features.pkl")

    return X[selected_features], selected_features, importance_df
