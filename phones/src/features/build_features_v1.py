import re
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import LabelEncoder

data = pd.read_json("../../data/raw-v1.json")
data.set_index("product_id", inplace=True)


def convert_string_to_list(mappings: set[tuple[str, str]], split_by: str, value):
    if pd.isnull(value):
        return list()

    for to_replace, replace_by in mappings:
        value = value.replace(to_replace, replace_by)

    lst = value.split(split_by)
    return lst


def encode_columns(data: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    df = data.copy()

    for col, encoding_type in columns.items():
        if encoding_type == "one_hot":
            column = df.pop(col)
            one_hot = pd.crosstab((s := column.explode()).index, s)
            new_col_names = [
                f"{''.join(map(lambda x: x[0], col.split('_'))) }_{val}"
                for val in one_hot.columns
            ]
            one_hot.columns = new_col_names
            df = df.join(one_hot)
        elif encoding_type == "label":
            le = LabelEncoder()
            df[f"{col}"] = le.fit_transform(df[col].astype(str))
            print(
                f"{col}: {list(enumerate(le.classes_))}",
            )

    return df


def combine_lists(column1: str, column2: str, row):
    # Combine the two lists and remove duplicates
    combined_list = [i for i in set(row[column1] + row[column2]) if i != ""]
    return combined_list


def fillna_by_manufacture(row_to_fill: str):
    for index, row in data[data[row_to_fill].isnull()].iterrows():
        manufacturer = row["Manufacturer"]
        manufacturer_df = data[data["Manufacturer"] == manufacturer]

        if len(manufacturer_df) <= 1:
            continue

        mode_result = manufacturer_df[row_to_fill].mode()
        if mode_result.empty:
            continue

        most_common_screen_type = mode_result.values[0]
        data.at[index, row_to_fill] = most_common_screen_type


def extract_warranty(row):
    warranty = row["Warranty Details"]

    if pd.isnull(warranty):
        return [0, 0]

    # Extract the warranty duration for the phone
    phone_duration = re.findall(r"(\d+)\s*year", warranty)
    phone_duration = int(phone_duration[0]) if phone_duration else 0

    # Extract the warranty duration for the items along with the phone
    items_duration = re.findall(r"(\d+)\s*months", warranty)
    items_duration = int(items_duration[0]) if items_duration else 0

    return [phone_duration, items_duration]


def simplify_cellular_technology(lst):
    if lst is None:
        return None

    replacements = {
        "4G LTE FDD": "4G LTE",
        "4G LTE TDD": "4G LTE",
        "3G WCDMA": "3G",
        "3 G": "3G",
        "4G LTE": "4G",
        "GSM": "2G",
        "LTE": "4G",
        "2G GSM": "2G",
        "4G VOLTE": "4G",
        "VOLTE": "4G",
    }

    return [replacements.get(val, val) for val in lst]


def extract_number(col_value):
    if pd.isnull(col_value):
        return None

    string = col_value.split()[0]
    return int(float(string))


def extract_grams(row_value):
    if pd.isnull(row_value):
        return None

    string, type = str(row_value).split()
    if type == "kg":
        string = float(string) * 1000
    string = int(float(string))
    return string


def extract_dimensions(row):
    dimensions = str(row["Item Dimensions"])

    if pd.isna(dimensions):
        return None

    seperations = [value.split()[0] for value in dimensions.split(" x ")]

    length, width, height = sorted(map(float, seperations), reverse=True)

    if height > 10:
        height /= 10

    if width > 10:
        width /= 10

    return [length, width, height]


def camera_features(desc):
    features = {}

    if pd.isna(desc):
        return None

    # Number of cameras
    if "quad" in desc.lower():
        features["camera_count"] = 4
    elif "triple" in desc.lower():
        features["camera_count"] = 3
    elif "dual" in desc.lower():
        features["camera_count"] = 2
    else:
        features["camera_count"] = 1  # default is one camera

    # Front and Rear camera
    if "front" in desc.lower():
        features["has_front_camera_details"] = int("mp" in desc.lower())
    else:
        features["has_front_camera_details"] = 0  # default is no specifics

    if "rear" in desc.lower():
        features["has_rear_camera_details"] = int("mp" in desc.lower())
    else:
        features["has_rear_camera_details"] = 0  # default is no specifics

    # Check for specific features
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

    # Camera resolution
    match = re.search(r"(\d+)MP", desc)
    if match:
        features["main_camera_MP"] = int(match.group(1))
    else:
        features["main_camera_MP"] = 12  # default is 12MP

    del features["has_rear_camera_details"], features["has_front_camera_details"]

    return pd.Series(features)


def extract_os_info(row):
    os_info = str(row["OS"])
    name = os_info.split()[0].lower()
    go_edition = int("go edition" in os_info)
    try:
        version = float(os_info.replace(",", "").split()[1])
    except:
        version = np.inf

    return name, version, go_edition


# remove columns where NA > 100

# Wireless network technology'
# GPU
# Imported By
# Item part number
# Colours displayed
# Phone Talk Time
# Phone Standby Time (with data)
# Package Dimensions
# Memory Storage Capacity
# Display technology
# Audio Jack
# Resolution
# Processor Brand
# Device interface - primary
# Processor Speed
# others...

na_counts = data.isna().sum().sort_values(ascending=False)
filtered_columns = na_counts[na_counts > 100].index.tolist()
data.drop(columns=filtered_columns, inplace=True)

to_remove = [
    "Batteries",
    "Item model number",
    "Product Dimensions",  # item dim better
    "Operating System",  # OS better,
    "Battery Power Rating",  # in mAh better
    "Whats in the box",  # i will use the proper english,
    "Other camera features",  # i will use Camera Description
    "Country of Origin",
    "Other display features",  # ITS ALL THE SAME
    "Model Name",
    "Brand",
    "Colour",
    "GPS",
    "Screen Size",
    "What's in the box",
    "Special features",
    "Network Service Provider",
]

data.drop(columns=to_remove, inplace=True, errors="ignore")

connectivity_tech_cols = [
    "Wireless communication technologies",
    "Connectivity technologies",
]

# for column in connectivity_tech_cols:
#     data[column] = data[column].apply(
#         partial(convert_string_to_list, {(";", " "), (",", " ")}, " ")
#     )

# data["connectivity_tech"] = data.apply(
#     partial(combine_lists, *connectivity_tech_cols), axis=1
# )
data.drop(columns=connectivity_tech_cols, inplace=True)

mapping = {
    "Samsung": "Samsung",
    "Redmi": "Xiaomi",
    "Xiaomi": "Xiaomi",
    "Xiaomi Technology India Private Limited": "Xiaomi",
    "Rising Stars Mobile India Private Limited": "Xiaomi",
    "OPPO Mobiles India Pvt Ltd": "OPPO",
    "Oppo Mobiles India Private Limited 5th Floor, Tower-B, Building No. 8, Haryana-122002, India": "OPPO",
    "Oppo Mobiles India Private Limited": "OPPO",
    "OPPO Mobiles India Private Limited": "OPPO",
    "Lava": "Lava",
    "LAVA": "Lava",
    "G mobiles": "G-Mobile",
    "G-MOBILE": "G-Mobile",
    "G Mobiles": "G-Mobile",
    "G-Mobile Devices Private Limited": "G-Mobile",
    "S MOBILE DEVICES PRIVATE LTD": "G-Mobile",
    "1 year manufacturer warranty for device and 6 months manufacturer warranty for in-box": "Generic",
    "Samsung India pvt Ltd": "Samsung",
    "Samsung India Electronics Pvt ltd": "Samsung",
    "Dixon Technologies (India) Ltd.,Plot No.6, Sector-90,Noida, Gautam Buddha Nagar, U.P. India-201305": "Dixon Technologies",
    "OnePlus": "OnePlus",
    "vivo Mobile India Pvt Ltd": "Vivo",
    "Vivo": "Vivo",
    "Vivo Mobile India Pvt Ltd": "Vivo",
    "vivo": "Vivo",
    "vivo Mobile India Private Limited": "Vivo",
    "For and on behalf of HMD Mobile India Private Limited": "Nokia",
    "Nokia": "Nokia",
    "Bhagwati Products Ltd": "Micromax",
    "MICROMAX": "Micromax",
    "Micromax": "Micromax",
    "Realme": "Realme",
    "iQOO": "iQOO",
    "Generic": "generic",
}

data["Manufacturer"] = data["Manufacturer"].replace(mapping)

fillna_by_manufacture("Screen Type")
fillna_by_manufacture("Battery type")
data[["Phone Warranty (months)", "Items Warranty (months)"]] = data.apply(
    extract_warranty, axis=1, result_type="expand"
)

# mapping = {
#     'Supports 5G* / 4G / 3G/ 2G *Supported 5G Bands: NR (SA & NSA): n1/n3/n5/n7/n8/n20/n28/n38/n40/n41/n66/n71/n75/n77/n78/n79  See more': "5G / 4G / 3G/ 2G",
#     "(4G,3G,2G)": "4G,3G,2G",
# }

# data['Cellular Technology'] = data['Cellular Technology'].replace(mapping)
# data["Cellular Technology"] = data["Cellular Technology"].apply(partial(convert_string_to_list, {(",", "/"), ("*", "/")}, '/')).apply(simplify_cellular_technology)

data["Inbuilt Storage (in GB)"] = data["Inbuilt Storage (in GB)"].apply(extract_number)
data["Battery Power (In mAH)"] = data["Battery Power (In mAH)"].apply(extract_number)
data["RAM"] = data["RAM"].apply(extract_number)
data["Item Weight"] = data["Item Weight"].apply(extract_grams)

data.dropna(
    subset=[
        "Inbuilt Storage (in GB)",
        "Item Weight",
        "RAM",
        "Item Dimensions",
        "Camera Description",
    ],
    inplace=True,
)

data["Inbuilt Storage (in GB)"] = data["Inbuilt Storage (in GB)"].astype(int)
data["RAM"] = data["RAM"].astype(int)
data["Item Weight"] = data["Item Weight"].astype(int)

data[["length", "width", "height"]] = data.apply(
    extract_dimensions, axis=1, result_type="expand"
)

camera_features_df = data["Camera Description"].apply(camera_features)
data = pd.concat([data, camera_features_df], axis=1)

replacements = {
    # "Android 12.0": ,
    # "Android 11.0": ,
    # "Android 10.0": ,
    # "Android 13.0": ,
    # "Android": ,
    # "Android 9.0": ,
    # "OxygenOS": ,
    # "MIUI 13": ,
    # "FunTouch OS 12": ,
    # "Android 8.1": ,
    "HiOS 7.6 based on Android 11": "HiOS 7.6",
    "Android 12(Go edition)": "Android 12, go edition",
    "HiOS 8.6 based on Android 12": "HiOS 8.6",
    "Funtouch OS 12 based on Android 12": "Funtouch OS 12",
    "Funtouch OS 12 (Based on Android 11)": "Funtouch OS 12",
    "MIUI 13, Android 12.0": "MIUI 13",
    "Funtouch OS 12 Based On Android 12": "Funtouch OS 12",
    # "Funtouch OS 12": ,
    "MIUI 13, Android 12": "MIUI 13",
    "Funtouch OS 13 Based On Android 13": "Funtouch OS 13",
    "MIUI 14, Android 13.0": "MIUI 14",
    "MIUI 12, Android 11.0": "MIUI 12",
    "MIUI 12.5, Android MIUI 12.5": "MIUI 12.5",
    "Android 11 MIUI 12.5 on, 3 years of Android updates, MIUI 12.5 on Android 11, 3 years of Android updates": "MIUI 12.5",
    # "Android 8.0": ,
    # "MIUI 12": ,
    "Android 11 - MiUI 12.5, MiUI 12.5 (Android 11)": "MiUI 12.5",
    "Android 11 Stock": "Android 11.0",
    "Funtouch OS 11 (Based on Android 11)": "Funtouch OS 11",
    "Funtouch OS 13 based on Android 13": "Funtouch OS 13",
    "Android 11, Funtouch OS 11.1": "Funtouch OS 11",
    "Go Edition, Android 11.0": "Android 11, go edition",
    "Funtouch OS 12 (Based on Android 12)": "Funtouch OS 12",
    # "MIUI 12.5": ,
    "MIUI 12, Android 10.0": "MIUI 12",
    "Android 11.1 based Funtouch OS 11.1": "Funtouch OS 11.1",
    "Android 11 - Funtouch OS 11.1, Funtouch OS 11.1": "Funtouch OS 11.1",
    "Android 11 MIUI 12.5": "MIUI 12.5",
    "HiOS 8.0 based on Android 11, Android 10.0": "HiOS 8.0",
    "HiOS 12.0 based on Android 12": "HiOS 12.0",
    # "Windows 11 Home": ,
    "Android 10 Go Edition": "Android 10, go edition",
}

data["OS"] = data["OS"].replace(replacements)
data[["os_name", "os_version", "os_go?"]] = data.apply(
    extract_os_info, axis=1, result_type="expand"
)

data = data.replace([np.inf, -np.inf], np.nan)
data["os_version"].fillna(data["os_version"].max() * 2, inplace=True)

data["Form factor"] = data["Form factor"].str.replace("Screen Touch", "Touch")
data["Form factor"] = (
    (data["Form factor"].str.lower().str.split().str[0].str.replace(",", ""))
    .replace("smartphones", "smartphone")
    .replace("touch", "touchscreen")
)

data.drop(
    columns=[
        "Warranty Details",
        "Manufacturer",
        "OS",
        "Camera Description",
        "Item Dimensions",
        "Cellular Technology",
    ],
    inplace=True,
)
data.dropna(inplace=True)

columns_to_encode = {
    # "connectivity_tech": "one_hot",
    "Form factor": "label",
    "Screen Type": "label",
    "Battery type": "label",
    "os_name": "label",
}


def label_smartphones(df):
    # Define the thresholds for success
    thresholds = {
        "RAM": 4,
        "Battery Power (In mAH)": 4000,
        "Phone Warranty (months)": 6,
        "Inbuilt Storage (in GB)": 64,
        "Average Rating": 4,
        "no_ratings": 1000,
        "camera_count": 2,
    }

    # Define the weightage for the Average Rating feature
    rating_weightage = 0.3
    no_rating_weightage = 0.5

    # Create a new column called "Label"
    df["is_success"] = 0

    df["no_ratings"] = df[
        ["no of 5 star", "no of 4 star", "no of 3 star", "no of 2 star", "no of 1 star"]
    ].sum(axis=1)

    # Calculate average rating
    df["Average Rating"] = (
        df["no of 5 star"] * 5
        + df["no of 4 star"] * 4
        + df["no of 3 star"] * 3
        + df["no of 2 star"] * 2
        + df["no of 1 star"]
    ) / df["no_ratings"]

    # Iterate over each row and apply the labeling logic
    for index, row in df.iterrows():
        percent = 0

        # Calculate percent based on the difference from the thresholds
        for feature, threshold in thresholds.items():
            if feature == "Average Rating":
                # Apply weightage to the Average Rating feature
                diff = row[feature] - threshold
                percent += rating_weightage * (
                    diff / 5
                )  # Scale the difference to [-1, 1] range
            if feature == "no_ratings":
                diff = row[feature] - threshold
                percent += no_rating_weightage * (diff / 5)
            elif row[feature] >= threshold:
                percent += 0.1
            else:
                percent -= 0.1

        # Assign the label based on the overall value of percent
        df.at[index, "is_success"] = int(percent >= 100)

    df.drop(
        columns=[
            "no_ratings",
        ],
        inplace=True,
    )

    return df


final = label_smartphones(encode_columns(data, columns_to_encode))

final.drop(
    columns=[
        "no of 5 star",
        "no of 4 star",
        "no of 3 star",
        "no of 2 star",
        "no of 1 star",
        "Average Rating",
    ],
    inplace=True,
)


final.to_csv("../../data/processed-v1.csv")

print(final.columns)
