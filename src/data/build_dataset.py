from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_data import load_raw_olist
from src.features.make_features import create_features


def build_orders_extended(raw_dir: str | Path,
                          processed_dir: str | Path) -> Path:
    """
    Construye y guarda el dataset maestro extendido 'orders_extended_master.csv'.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw_olist(raw_dir)

    orders = data["orders"].copy()
    customers = data["customers"].copy()
    order_items = data["order_items"].copy()
    order_payments = data["order_payments"].copy()
    reviews = data["order_reviews"].copy()
    products = data["products"].copy()
    sellers = data["sellers"].copy()
    cat_trad = data["cat_trad"].copy()
    geo = data["geolocation"].copy()

    # Target extendido
    orders["is_canceled_strict"] = (orders["order_status"] == "canceled").astype(int)

    cond_canceled = orders["order_status"] == "canceled"
    cond_unavailable = orders["order_status"] == "unavailable"

    last_purchase = orders["order_purchase_timestamp"].max()
    umbral_fecha = last_purchase - pd.Timedelta(days=30)

    cond_pending_old = (
        orders["order_status"].isin(["created", "processing"])
        & orders["order_delivered_customer_date"].isna()
        & (orders["order_purchase_timestamp"] < umbral_fecha)
    )

    orders["order_canceled_extended"] = np.where(
        cond_canceled | cond_unavailable | cond_pending_old,
        1,
        0,
    )

    # Master table
    df = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(order_items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(order_payments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(cat_trad, on="product_category_name", how="left")
    )

    # Features
    df = create_features(df, order_items=order_items, geolocation=geo)

    out_path = processed_dir / "orders_extended_master.csv"
    df.to_csv(out_path, index=False)

    return out_path
