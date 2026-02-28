import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_xgb(X_train, y_train):
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs), probs


def save_model(model, path):
    joblib.dump(model, path)