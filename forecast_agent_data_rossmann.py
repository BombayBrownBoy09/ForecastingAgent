import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_prep_data():
    sales = pd.read_csv("train.csv", parse_dates=["Date"])
    stores = pd.read_csv("store.csv")
    df = sales.merge(stores, on="Store", how="left")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    df.rename(
        columns={
            "Date": "date",
            "Store": "tcin",
            "Sales": "units_sold",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday"
        },
        inplace=True
    )

    from sklearn.preprocessing import LabelEncoder
    le_state = LabelEncoder()
    le_school = LabelEncoder()
    df["state_holiday_id"] = le_state.fit_transform(df["state_holiday"].astype(str))
    df["school_holiday_id"] = le_school.fit_transform(df["school_holiday"].astype(str))

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    le_region   = LabelEncoder()
    le_category = LabelEncoder()
    df["region_id"] = le_region.fit_transform(df["store_type"])
    df["category_id"] = le_category.fit_transform(df["assortment"])

    df["new_launch_flag"] = 0

    agg_df = (
        df[[
            "date", "tcin", "region_id", "category_id", "new_launch_flag",
            "day_of_week", "week_of_year", "month", "promo",
            "state_holiday_id", "school_holiday_id", "units_sold"
        ]]
        .groupby(
            [
                "date", "tcin", "region_id", "category_id", "new_launch_flag",
                "day_of_week", "week_of_year", "month", "promo",
                "state_holiday_id", "school_holiday_id"
            ],
            as_index=False
        )
        .agg({"units_sold": "sum"})
    )
    agg_df.sort_values(by=["tcin", "date"], inplace=True)
    agg_df.reset_index(drop=True, inplace=True)
    return agg_df
