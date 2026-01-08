import numpy as np

def handle_skewness(df, numeric_cols, threshold=1.0):
    skewed_cols = df[numeric_cols].skew()
    high_skew_cols = skewed_cols[abs(skewed_cols) > threshold].index.tolist()

    print("Highly Skewed Columns:", high_skew_cols)

    for col in high_skew_cols:
        min_val = df[col].min()

        if min_val <= 0:
            df[col] = np.log1p(df[col] - min_val)
        else:
            df[col] = np.log1p(df[col])

    return df, high_skew_cols
