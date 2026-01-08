from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_model(X, y):
    # 🔹 STEP A: Sample data for tuning (CRITICAL)
    X_sample = X.sample(20000, random_state=42)
    y_sample = y.loc[X_sample.index]

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=5,          # 🔥 reduce iterations
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,          # 🔥 VERY IMPORTANT
        verbose=1,
        random_state=42
    )

    # 🔹 STEP B: Tune on SAMPLE
    search.fit(X_sample, y_sample)

    print("Best Params:", search.best_params_)

    # 🔹 STEP C: Retrain best model on FULL DATA
    best_model = search.best_estimator_
    best_model.n_jobs = 1  # prevent memory crash
    best_model.fit(X, y)

    return best_model
