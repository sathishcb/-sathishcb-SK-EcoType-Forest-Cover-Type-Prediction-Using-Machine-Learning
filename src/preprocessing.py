from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df, target_col="Cover_Type"):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # ✅ SAFE DROP (no error if column missing)
    numeric_cols = numeric_cols.drop(target_col, errors="ignore")

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col])

    joblib.dump(encoder, "label_encoder.pkl")

    return df, numeric_cols
