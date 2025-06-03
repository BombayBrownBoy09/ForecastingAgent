# Generate a Python script that loads and prepares sales data for Rossmann data forecasting. It should:
# 1. Import pandas, numpy, and LabelEncoder from scikit-learn.
# 2. Define a function load_and_prep_data() that reads "train.csv" (with "Date" parsed as dates) and "store.csv".
# 3. Merge the two DataFrames on "Store" using a left join.
# 4. Filter the data to include only rows where "Open" is 1 and "Sales" > 0.
# 5. Rename columns: "Date" to "date", "Store" to "tcin", "Sales" to "units_sold", "StoreType" to "store_type", "Assortment" to "assortment", "Promo" to "promo", "StateHoliday" to "state_holiday", and "SchoolHoliday" to "school_holiday".
# 6. Encode "state_holiday" and "school_holiday" as numeric columns "state_holiday_id" and "school_holiday_id" using LabelEncoder.
# 7. Extract "day_of_week", "week_of_year" (from dt.isocalendar), and "month" from the "date" column.
# 8. Encode "store_type" to "region_id" and "assortment" to "category_id" using LabelEncoder.
# 9. Set a new column "new_launch_flag" to 0.
# 10. Create an aggregated DataFrame (agg_df) by grouping on:
#     ["date", "tcin", "region_id", "category_id", "new_launch_flag", "day_of_week", "week_of_year", "month", "promo", "state_holiday_id", "school_holiday_id"]
#     and summing "units_sold". Sort by "tcin" and "date", reset the index, and return agg_df.

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
