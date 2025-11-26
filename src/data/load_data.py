from pathlib import Path
import pandas as pd

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "cat_trad": "product_category_name_translation.csv",
}


def load_raw_olist(raw_dir: str | Path) -> dict:
    """
    Carga todos los CSV originales de Olist desde data/raw.
    """
    raw_dir = Path(raw_dir)
    data = {}

    data["customers"] = pd.read_csv(raw_dir / RAW_FILES["customers"])
    data["geolocation"] = pd.read_csv(raw_dir / RAW_FILES["geolocation"])
    data["order_items"] = pd.read_csv(raw_dir / RAW_FILES["order_items"])
    data["order_payments"] = pd.read_csv(raw_dir / RAW_FILES["order_payments"])
    data["order_reviews"] = pd.read_csv(raw_dir / RAW_FILES["order_reviews"])
    data["products"] = pd.read_csv(raw_dir / RAW_FILES["products"])
    data["sellers"] = pd.read_csv(raw_dir / RAW_FILES["sellers"])
    data["cat_trad"] = pd.read_csv(raw_dir / RAW_FILES["cat_trad"])

    data["orders"] = pd.read_csv(
        raw_dir / RAW_FILES["orders"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    return data
