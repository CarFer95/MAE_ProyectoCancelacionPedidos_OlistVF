from __future__ import annotations

import numpy as np
import pandas as pd


def _add_purchase_ym(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["purchase_ym"] = (
        df["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
        .str.replace("-", "")
    )
    return df


def _remove_leakage(X: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = [
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp",
        "review_score",
        "review_comment_message",
        "has_review_comment",
        "review_comment_length",
        "review_creation_delay_days",
        "review_answer_delay_days",
    ]
    leakage_cols = [c for c in leakage_cols if c in X.columns]
    X = X.drop(columns=leakage_cols, errors="ignore")
    return X


def _replace_inf_with_nan(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)


def prepare_xy(df_master: pd.DataFrame,
               target_col: str = "order_canceled_extended"):
    df = df_master.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df = _add_purchase_ym(df)

    features_drop = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "is_canceled_strict",
        target_col,
    ]

    train_months = [
        "201610", "201611", "201612",
        "201701", "201702", "201703", "201704", "201705", "201706",
        "201707", "201708", "201709", "201710", "201711", "201712",
        "201801", "201802", "201803", "201804",
    ]
    backtest_months = ["201805", "201806", "201807"]
    final_test_month = ["201808"]

    df_train = df[df["purchase_ym"].isin(train_months)]
    df_backtest = df[df["purchase_ym"].isin(backtest_months)]
    df_final = df[df["purchase_ym"].isin(final_test_month)]

    X_train = df_train.drop(columns=features_drop + ["purchase_ym"], errors="ignore")
    y_train = df_train[target_col].astype(int)

    X_backtest = df_backtest.drop(columns=features_drop + ["purchase_ym"], errors="ignore")
    y_backtest = df_backtest[target_col].astype(int)

    X_final = df_final.drop(columns=features_drop + ["purchase_ym"], errors="ignore")
    y_final = df_final[target_col].astype(int)

    X_train = _remove_leakage(X_train)
    X_backtest = _remove_leakage(X_backtest)
    X_final = _remove_leakage(X_final)

    X_train = _replace_inf_with_nan(X_train)
    X_backtest = _replace_inf_with_nan(X_backtest)
    X_final = _replace_inf_with_nan(X_final)

    return X_train, y_train, X_backtest, y_backtest, X_final, y_final
