import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_battery_type(value):
    lower_value = str(value).lower()

    if "polymer" in lower_value or "phosphate" in lower_value:
        return "Lithium Polymer"

    if "ion" in lower_value or "cobalt" in lower_value:
        return "Lithium Ion"

    return np.nan


def fillna_by_manufacturer(row_to_fill, data):
    for index, row in data[data[row_to_fill].isnull()].iterrows():
        manufacturer = row["manufacturer"]
        manufacturer_df = data[data["manufacturer"] == manufacturer]

        if len(manufacturer_df) <= 1:
            continue

        mode_result = manufacturer_df[row_to_fill].mode()
        if mode_result.empty:
            continue

        most_common_screen_type = mode_result.values[0]
        data.at[index, row_to_fill] = most_common_screen_type


def extract_number(col_value):
    if pd.isnull(col_value):
        return None

    string = col_value.split()[0]
    return int(float(string))


def extract_grams(row_value):
    if pd.isnull(row_value):
        return None

    string, unit = str(row_value).split()
    if unit == "kg":
        string = float(string) * 1000
    string = int(float(string))
    return string


def extract_dimensions(row):
    dimensions = str(row["dimensions"])

    if pd.isna(dimensions):
        return None

    separations = [value.split()[0] for value in dimensions.split(" x ")]
    length, width, height = sorted(map(float, separations), reverse=True)

    if height >= 4:
        height /= 10

    if width > 15:
        width /= 10

    if length > 25:
        length /= 10

    return [length, width, height]


def extract_os_info(row):
    os_info = str(row["os"]).lower()

    # Split the os info by a comma, strip each item, and remove duplicates
    os_items = list(set([item.strip() for item in os_info.split(",")]))

    raw_name = os_items[0].strip()

    for item in os_items:
        if "android" not in item or "based on android" in item or "on android" in item:
            raw_name = item.strip()

    try:
        match = re.search(r"\d+(\.\d+)?", raw_name)
        assert match
        version = float(match.group())
        name = raw_name[: raw_name.find(str(int(version)))].strip()
    except:
        version = np.inf
        name = os_info.split()[0]

    if "android" in name:
        name = name.split()[0]

    return name, version


def extract_phone_warranty(value):
    duration = 0  # Default value

    match = re.search(r"(\d+)\s*(year|month|yr|mo)", str(value), re.IGNORECASE)
    if match:
        duration_value = int(match.group(1))
        duration_unit = match.group(2).lower()

        if duration_unit in ["year", "yr"]:
            duration = duration_value * 12
        elif duration_unit in ["month", "mo"]:
            duration = duration_value

    return duration


def camera_features(desc):
    features = {}

    if pd.isna(desc):
        return None

    features["camera_count"] = 2

    if "quad" in desc.lower():
        features["camera_count"] += 4
    if "triple" in desc.lower():
        features["camera_count"] += 3
    if "dual" in desc.lower():
        features["camera_count"] += 2

    features["has_front_camera_details"] = (
        int("mp" in desc.lower()) if "front" in desc.lower() else 0
    )
    features["has_rear_camera_details"] = (
        int("mp" in desc.lower()) if "rear" in desc.lower() else 0
    )

    features["cam_has_AI"] = int(
        "ai" in desc.lower() or features["has_rear_camera_details"]
    )
    features["cam_has_OIS"] = int(
        "ois" in desc.lower() or features["has_rear_camera_details"]
    )
    features["cam_has_Zoom"] = int(
        "zoom" in desc.lower() or features["has_rear_camera_details"]
    )
    features["cam_has_HDR"] = int(
        "hdr" in desc.lower() or features["has_rear_camera_details"]
    )
    features["cam_has_Macro"] = int(
        "macro" in desc.lower() or features["has_rear_camera_details"]
    )
    features["cam_has_Portrait"] = int(
        "portrait" in desc.lower() or features["has_rear_camera_details"]
    )

    match = re.search(r"(\d+)MP", desc)
    features["main_camera_MP"] = int(match.group(1)) if match else 12

    del features["has_rear_camera_details"], features["has_front_camera_details"]

    return pd.Series(features)


def label_smartphones(df):
    thresholds = {
        "ram": 4,
        "battery_power": 4000,
        "phone_warranty (months)": 6,
        "inbuilt_storage": 64,
        "no_ratings": 1000,
        "camera_count": 2,
        "avg_rating": 4,
    }

    rating_weightage = 0.3
    no_rating_weightage = 0.5

    df["is_success"] = False
    df["percentage"] = 0

    df["no_ratings"] = df[
        ["no of 5 star", "no of 4 star", "no of 3 star", "no of 2 star", "no of 1 star"]
    ].sum(axis=1)

    df["avg_rating"] = (
        df["no of 5 star"] * 5
        + df["no of 4 star"] * 4
        + df["no of 3 star"] * 3
        + df["no of 2 star"] * 2
        + df["no of 1 star"]
    ) / df["no_ratings"]

    for index, row in df.iterrows():
        percent = 0

        for feature, threshold in thresholds.items():
            if feature == "avg_rating":
                diff = row[feature] - threshold
                percent += rating_weightage * (diff / 5)
            if feature == "no_ratings":
                diff = row[feature] - threshold
                percent += no_rating_weightage * (diff / 5)
            elif row[feature] >= threshold:
                percent += 0.1
            else:
                percent -= 0.1

        df.at[index, "is_success"] = percent >= 100
        df.at[index, "percentage"] = percent

    df.drop(
        columns=[
            "no of 5 star",
            "no of 4 star",
            "no of 3 star",
            "no of 2 star",
            "no of 1 star",
            "avg_rating",
            "no_ratings",
            "percentage",
        ],
        inplace=True,
    )

    return df


def encode_columns(data, columns):
    df = data.copy()

    for col, encoding_type in columns.items():
        if encoding_type == "one_hot":
            column = df.pop(col)
            one_hot = pd.get_dummies(column, prefix=col)
            df = pd.concat([df, one_hot], axis=1)
        elif encoding_type == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"{col}: {list(enumerate(le.classes_))}")

    return df


def main():
    data = pd.read_json("../../data/raw-v2.json")
    data.set_index("product_id", inplace=True)

    data.drop_duplicates(inplace=True)

    data = data[
        [
            "mrp",
            "model_name",
            "no of 5 star",
            "no of 4 star",
            "no of 3 star",
            "no of 2 star",
            "no of 1 star",
            "os",
            "ram",
            "inbuilt_storage",
            "dimensions",
            "weight",
            "battery_power",
            "battery_type",
            "camera",
            "warranty",
            "form_factor",
            "manufacturer",
        ]
    ]

    data[
        ["no of 5 star", "no of 4 star", "no of 3 star", "no of 2 star", "no of 1 star"]
    ] = data[
        ["no of 5 star", "no of 4 star", "no of 3 star", "no of 2 star", "no of 1 star"]
    ].fillna(
        0
    )

    data["manufacturer"] = data["manufacturer"].str.lower().str.split().str[0]
    data["manufacturer"] = data["manufacturer"].str.replace("s", "s-mobile")
    data["manufacturer"] = data["manufacturer"].str.replace("g", "g-mobile")

    data["form_factor"] = data["form_factor"].str.replace("Screen Touch", "Touch")
    data["form_factor"] = (
        (data["form_factor"].str.lower().str.split().str[0].str.replace(",", ""))
        .replace("smartphones", "smartphone")
        .replace("touch", "touchscreen")
    )

    data["battery_type"] = data["battery_type"].apply(clean_battery_type)

    fillna_by_manufacturer("battery_type", data)

    data.dropna(subset=["battery_type", "form_factor"], inplace=True)

    data["battery_power"] = data["battery_power"].apply(extract_number).astype(int)
    data["inbuilt_storage"] = data["inbuilt_storage"].apply(extract_number).astype(int)
    data["ram"] = data["ram"].apply(extract_number).astype(int)
    data["weight"] = data["weight"].apply(extract_grams)

    data[["length", "width", "height"]] = data.apply(
        extract_dimensions, axis=1, result_type="expand"
    )

    data[["os_name", "os_version"]] = data.apply(  # , "os_go?"
        extract_os_info, axis=1, result_type="expand"
    )
    data["os_name"] = data["os_name"].replace("s", "s30+")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["os_version"].fillna(data["os_version"].max() * 2, inplace=True)

    data["phone_warranty (months)"] = data["warranty"].apply(extract_phone_warranty)

    camera_features_df = data["camera"].apply(camera_features)
    data = pd.concat([data, camera_features_df], axis=1)

    data.drop(
        columns=[
            "model_name",
            "dimensions",
            "manufacturer",
            "warranty",
            "camera",
            "os",
        ],
        inplace=True,
    )

    columns_to_encode = {
        "battery_type": "label",
        "form_factor": "label",
        "os_name": "label",
    }

    data = encode_columns(data, columns_to_encode)

    data = label_smartphones(data)

    data.to_csv("../../data/processed-v2.csv")
    print(data.columns)
    print(data["is_success"].value_counts())


if __name__ == "__main__":
    main()
