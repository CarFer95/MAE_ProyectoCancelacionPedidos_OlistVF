from __future__ import annotations

from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

ModelType = Literal["logreg", "rf", "xgb"]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def _build_model(model_type: ModelType):
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
    if model_type == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=3.0,
            tree_method="hist",
            random_state=42,
        )
    raise ValueError(f"Modelo no soportado: {model_type}")


def create_model_pipeline(X_sample: pd.DataFrame,
                          model_type: ModelType = "xgb") -> Pipeline:
    preprocessor = _build_preprocessor(X_sample)
    model = _build_model(model_type)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return clf
