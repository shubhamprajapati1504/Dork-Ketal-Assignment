"""Create and populate the SQLite database used for the assignment."""

import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Data" / "processed_csv_dataset.csv"
DATABASE_PATH = BASE_DIR / "database" / "inventory.db"

SQLITE_COLUMNS = {
    "Date": "TEXT",
    "Store ID": "TEXT",
    "Product ID": "TEXT",
    "Category": "TEXT",
    "Region": "TEXT",
    "Inventory Level": "INTEGER",
    "Units Sold": "INTEGER",
    "Units Ordered": "INTEGER",
    "Price": "REAL",
    "Discount": "REAL",
    "Weather Condition": "TEXT",
    "Promotion": "INTEGER",
    "Competitor Pricing": "REAL",
    "Seasonality": "TEXT",
    "Epidemic": "INTEGER",
    "Demand": "INTEGER",
    "year": "INTEGER",
    "month": "INTEGER",
    "day": "INTEGER",
    "day_of_week": "INTEGER",
    "week_of_year": "INTEGER",
    "is_weekend": "INTEGER",
    "dow_sin": "REAL",
    "dow_cos": "REAL",
    "month_sin": "REAL",
    "month_cos": "REAL",
}


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier, including CSV column names containing spaces."""
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def create_database() -> None:
    """Replace the sales table with the latest processed CSV data."""
    data = pd.read_csv(CSV_PATH)
    DATABASE_PATH.parent.mkdir(exist_ok=True)

    column_definitions = ",\n        ".join(
        f"{quote_identifier(name)} {data_type}" for name, data_type in SQLITE_COLUMNS.items()
    )
    create_table_sql = f"""
    CREATE TABLE sales_data (
        {column_definitions}
    )
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS sales_data")
        cursor.execute(create_table_sql)
        data.to_sql("sales_data", connection, if_exists="append", index=False)
        cursor.execute(
            'CREATE INDEX idx_sales_product_date ON sales_data ("Product ID", "Date")'
        )
        row_count = cursor.execute("SELECT COUNT(*) FROM sales_data").fetchone()[0]

    print(f"Created {DATABASE_PATH} with {row_count:,} sales records.")


if __name__ == "__main__":
    create_database()
