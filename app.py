import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import urllib.request


# Load the random forest model
model_path_1 = "./phones/src/models/best_model.joblib"
model_1 = joblib.load(model_path_1)

model_path_2 = "./laptops/models/random_forest.joblib"
model_2 = joblib.load(model_path_2)

print("Model paths:")
print("Model 1:", model_path_1)
print("Model 2:", model_path_2)

if os.path.exists(model_path_1):
    model_1 = joblib.load(model_path_1)
    print("Model 1 loaded successfully.")
else:
    print("Error: Model 1 file not found.")

if os.path.exists(model_path_2):
    model_2 = joblib.load(model_path_2)
    print("Model 2 loaded successfully.")
else:
    print("Error: Model 2 file not found.")


# Streamlit app starts here
# ... (Your input fields and UI elements...)



# CSS classes for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .section {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: bold;
    }

    .expander-content {
        padding-top: 10px;
    }

    button.edgvbvh9 {
        background-color: #F63366;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }

    button.edgvbvh9:hover {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

device_type = st.select_slider(
    "What do you wish to predict?",
    options=["phones", "laptops"],
)

if device_type == "phones":
    st.markdown("<div class='title'>Phone Success Rate</div>", unsafe_allow_html=True)
    st.text("*Fill Every Section")

    # encodings
    battery_type_e = [(0, "Lithium Ion"), (1, "Lithium Polymer")]
    battery_type_d = {label: value for value, label in battery_type_e}

    form_factor_e = [
        (0, "bar"),
        (1, "flip"),
        (2, "foldable"),
        (3, "palm"),
        (4, "phablet"),
        (5, "slate"),
        (6, "slider"),
        (7, "smartphone"),
        (8, "touchscreen"),
    ]
    form_factor_d = {label: value for value, label in form_factor_e}

    os_name_e = [
        (0, "android"),
        (1, "funtouch os"),
        (2, "hios"),
        (3, "miui"),
        (4, "oxygenos"),
        (5, "s30+"),
        (6, "windows"),
    ]
    os_name_d = {label: value for value, label in os_name_e}

    st.markdown("<div class='section'>General</div>", unsafe_allow_html=True)
    mrp = st.number_input("MRP", value=0, step=1, format="%d")
    ram = st.number_input("RAM", value=0, step=1, format="%d")
    phone_warranty = st.number_input(
        "Phone Warranty (Months)",
        value=0,
        step=1,
        format="%d",
    )
    inbuilt_storage = st.number_input("Inbuilt Storage", value=0, step=1, format="%d")
    form_factor = st.selectbox(
        "Form Factor",
        tuple(map(lambda x: x[-1], form_factor_e)),
    )

    st.markdown("<div class='section'>Dimensions</div>", unsafe_allow_html=True)
    length = st.number_input(
        "Length",
        value=0.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.1f",
    )
    width = st.number_input(
        "Width",
        value=0.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.1f",
    )
    height = st.number_input(
        "Height",
        value=0.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.1f",
    )
    weight = st.number_input("Weight", value=0, step=1, format="%d")

    st.markdown("<div class='section'>Operating System</div>", unsafe_allow_html=True)
    os_name = st.selectbox(
        "OS Name",
        tuple(map(lambda x: x[-1], os_name_e)),
    )
    os_version = st.number_input(
        "OS Version",
        value=0.0,
        min_value=0.0,
        step=0.1,
        format="%.1f",
    )

    st.markdown("<div class='section'>Camera Features</div>", unsafe_allow_html=True)
    camera_count = st.number_input(
        "Total Number of Individual Cameras",
        value=0,
        step=1,
        min_value=0,
        format="%d",
    )
    cam_has_AI = st.checkbox("Camera has AI")
    cam_has_OIS = st.checkbox("Camera has OIS")
    cam_has_zoom = st.checkbox("Camera has Zoom")
    cam_has_hdr = st.checkbox("Camera has HDR")
    cam_has_macro = st.checkbox("Camera has Macro")
    cam_has_portrait = st.checkbox("Camera has Portrait")
    main_cam_mp = st.checkbox("Camera MP")

    st.markdown("<div class='section'>Battery</div>", unsafe_allow_html=True)
    battery_power = st.number_input(
        "Battery Power (in mAh)",
        value=0,
        step=1,
        format="%d",
    )
    battery_type = st.radio("Battery Type", tuple(map(lambda x: x[-1], battery_type_e)))

# Function to show graphs from specific directories
def show_graphs(directory_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, directory_path)
    file_list = os.listdir(image_dir)

    if len(file_list) == 0:
        st.warning("No graphs found in this category.")
    else:
        # Calculate the number of images per row
        num_images_per_row = 2
        num_images = len(file_list)
        num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

        # Set the space between images
        image_spacing = 20

        idx = 0
        for row in range(num_rows):
            cols = st.columns(num_images_per_row)
            for col in cols:
                if idx < num_images:
                    filename = file_list[idx]
                    image_path = os.path.join(image_dir, filename)
                    image = Image.open(image_path)
                    col.image(image, caption=filename, use_column_width=True)
                    idx += 1
                else:
                    col.empty()  # Create an empty space if there are no more images to display




if st.button("Predict It"):
    if (
        mrp <= 0
        or ram <= 0
        # or phone_warranty <= 0
        or inbuilt_storage <= 0
        or length <= 0
        or width <= 0
        or height <= 0
        or weight <= 0
        or os_version <= 0
        or camera_count < 0
        or battery_power <= 0
    ):
        st.error("Invalid input. Please ensure all numeric values are greater than zero.")
    elif "" in [form_factor, os_name, battery_type]:
        st.error("Invalid input. Please Select a valid Value for the Select Fields")
    else:
        # Prepare the data for prediction
        data = [
            mrp,
            ram,
            inbuilt_storage,
            weight,
            battery_power,
            battery_type_d[battery_type],
            form_factor_d[form_factor],
            length,
            width,
            height,
            os_name_d[os_name],
            os_version,
            phone_warranty,
            camera_count,
            cam_has_AI,
            cam_has_OIS,
            cam_has_zoom,
            cam_has_hdr,
            cam_has_macro,
            cam_has_portrait,
            main_cam_mp,
        ]
        # Convert the data to numpy array and reshape it
        data = np.array(data).reshape(1, -1)

        # Add print statements for debugging
        print("Data for phone prediction:")
        print(data)

        # Make prediction using model_1
        prediction = model_1.predict(data)
        print("Prediction for phone:")
        print(prediction)

        if prediction[0]:
            st.success("Product should be a success in the current Market")
        else:
            st.warning("Product will be a failure in current Market")
            # Display prediction result with a pie chart

# Description of the dataset
st.markdown("""
The following button allows you to view and download the phone dataset on which the models are trained. 
Click on the button to download the dataset.
""")

# Function to load and display the dataset
def view_dataset():
    df = pd.read_csv("processed-v2.csv")
    st.dataframe(df)



# File upload to load the dataset
uploaded_file = st.file_uploader("Upload the dataset (processed-v2.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# Button to view the dataset
if st.button("View Dataset by downloading from the github and drop in here"):
    view_dataset()

# Link to view the dataset on GitHub
github_link = "https://github.com/laxmanprasadsomarajutus/myproject/blob/main/phones/data/processed-v2.csv"
if st.button("View Dataset on GitHub"):
    st.markdown(f"Click [here]({github_link}) to view the dataset on GitHub.")


# Create a section for all graphs
st.subheader("Graphs for Analysis")

# Description of the graphs
st.markdown("""
This section contains graphs that have been generated based on the designed, tested, and training data. 
If you would like to view the graphs, click the button below:
""")
            
# Define a flag variable to track if graphs are being displayed
graphs_displayed = False

# Button to view the graphs
if not graphs_displayed and st.button("View Graphs"):
    with st.expander("Correlation Graphs", expanded=True):
        show_graphs("phones/src/plots/correlation")

    with st.expander("Correlation with Success Graphs", expanded=True):
        show_graphs("phones/src/plots/correlationwithsuccess")

    with st.expander("Distributions Graphs", expanded=True):
        show_graphs("phones/src/plots/distributions")

    with st.expander("Success Rate by Category Graphs", expanded=True):
        show_graphs("phones/src/plots/successratebycategory")

    with st.expander("Success vs Features Graphs", expanded=True):
        show_graphs("phones/src/plots/successvsfeatures")

    # Set the flag to True to indicate that graphs are displayed
    graphs_displayed = True

# Add a back button to return to the main UI if graphs are displayed
if graphs_displayed and st.button("Back"):
    graphs_displayed = False

elif device_type == "laptops":
    st.markdown("<div class='title'>Laptops Success Rate</div>", unsafe_allow_html=True)
    st.text("*Fill Every Section")

    os_e = {
        "chrome os": 0,
        "dos": 1,
        "mac os": 2,
        "windows": 3,
        "windows 10": 4,
        "windows 11": 5,
        "windows 7": 6,
    }
    hard_disk_type_e = {"HDD": 0, "Hybrid": 1, "SSD": 2, "eMMC": 3}
    processor_brand_e = {"AMD": 0, "IBM": 1, "Intel": 2, "MediaTek": 3, "NVIDIA": 4}
    display_type_e = {
        "AMOLED": 0,
        "FHD": 1,
        "LCD": 2,
        "LED": 3,
        "OLED": 4,
        "Pixel Sense": 5,
    }
    form_factor_e = {
        "Chromebook": 0,
        "Clamshell": 1,
        "Convertible": 2,
        "Gaming Laptop": 3,
        "Laptop": 4,
        "Netbook": 5,
        "Notebook": 6,
        "Thin & Light": 7,
        "Thin and Light": 8,
        "Ultra-Portable": 9,
    }
    hard_drive_size_unit_e = {"gb": 0, "tb": 1}
    battery_type_e = {
        "Lithium Ion": 0,
        "Lithium Metal": 1,
        "Lithium Polymer": 2,
        "nan": 3,
    }

    mrp = st.number_input("MRP", value=0, step=1, format="%d")

    os = st.selectbox(
        "OS",
        os_e.keys(),
    )
    hard_disk_type = st.selectbox(
        "Hard Disk Type",
        hard_disk_type_e.keys(),
    )

    hard_drive_size = st.number_input("Hard Drive Size", value=0, step=1, format="%d")
    hard_drive_unit = st.selectbox(
        "Hard Drive Unit?",
        hard_drive_size_unit_e.keys(),
    )

    ram_memory = st.number_input("Ram Memory in GB", value=0, step=1, format="%d")

    display_type = st.selectbox(
        "Display Type",
        display_type_e.keys(),
    )

    form_factor = st.selectbox(
        "Form Factor",
        form_factor_e.keys(),
    )

    length = st.number_input(
        "Length",
        value=0.0,
        min_value=0.0,
        step=0.1,
        format="%.1f",
    )
    width = st.number_input(
        "Width",
        value=0.0,
        min_value=0.0,
        step=0.1,
        format="%.1f",
    )
    height = st.number_input(
        "Height",
        value=0.0,
        min_value=0.0,
        step=0.1,
        format="%.1f",
    )

    weight = st.number_input(
        "Weight (kg)",
        value=0.00,
        min_value=0.00,
        step=0.01,
        format="%.2f",
    )

    screen_res_w = st.number_input("Screen Res W", value=1920, step=20, format="%d")
    screen_res_h = st.number_input("Screen Res H", value=1080, step=20, format="%d")

    processor_brand = st.selectbox(
        "Process Brand",
        processor_brand_e.keys(),
    )
    processor_count = st.number_input("Processor Count", value=0, step=1, format="%d")
    battery_type = st.selectbox(
        "Battery Type",
        battery_type_e.keys(),
    )

    if st.button("Predict It"):
        if (
            mrp <= 0
            or ram_memory <= 0
            # or phone_warranty <= 0
            or processor_count <= 0
            or length <= 0
            or width <= 0
            or height <= 0
            or weight <= 0
            or screen_res_w <= 0
            or screen_res_h < 0
            or hard_drive_size <= 0
        ):
            st.error(
                "Invalid input. Please ensure all numeric values are greater than zero."
            )
        elif "" in [
            os,
            hard_disk_type,
            processor_brand,
            display_type,
            form_factor,
            hard_drive_unit,
            battery_type,
        ]:
            st.error("Invalid input. Please Select a valid Value for the Select Fields")
        else:
            # Prepare the data for prediction
            try:
                data = [
                    mrp,
                    os_e[os],
                    hard_disk_type_e[hard_disk_type],
                    ram_memory,
                    processor_brand_e[processor_brand],
                    processor_count,
                    display_type_e[display_type],
                    form_factor_e[form_factor],
                    screen_res_w,
                    screen_res_h,
                    length,
                    width,
                    height,
                    weight,
                    hard_drive_size,
                    hard_drive_size_unit_e[hard_drive_unit],
                    battery_type_e[battery_type],
                ]
            except KeyError:
                st.error("Contact Dev as there is issue with code")
            else:
                # Convert the data to numpy array and reshape it
                data = np.array(data).reshape(1, -1)

                # Make prediction
                prediction = model_2.predict(data)
                if prediction is True:
                    st.success(f"Product should be a success in the current Market")
                else:
                    st.warning(f"Product will be a failure in current Market")
