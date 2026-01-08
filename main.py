from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.skewness_handler import handle_skewness
from src.imbalance_handler import balance_data
from src.feature_selection import select_features
from src.model_training import train_model
from src.evaluation import evaluate_model

from sklearn.model_selection import train_test_split
import joblib

# Load data
df = load_data("data/cover_type.csv")

# Preprocessing
df, numeric_cols = preprocess_data(df)

# Automatic skewness handling
df, skewed_cols = handle_skewness(df, numeric_cols)

# Split
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
X_res, y_res = balance_data(X_train, y_train)

# Feature selection
X_selected, selected_features, importance_df = select_features(X_res, y_res)

X_test_selected = X_test[selected_features]

# Train model
best_model = train_model(X_selected, y_res)

# Evaluate
evaluate_model(best_model, X_test_selected, y_test)

# Save model
joblib.dump(best_model, "forest_cover_model.pkl")

print("✅ Project pipeline completed successfully")
