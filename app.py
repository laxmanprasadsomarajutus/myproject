import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import webbrowser
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import urllib.request



# Load the random forest model
model_path_1 = "./phones/src/models/best_model_final.joblib"
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


# Function to open the link in a new tab
def open_link_in_new_tab(link, text):
    new_tab_code = f'<a href="{link}" target="_blank" style="font-size: 16px; color: white;">{text}</a>'
    st.markdown(new_tab_code, unsafe_allow_html=True)

def show_developer_details():
    st.markdown(
        """
        <div style="border: 1px solid #ccc; padding: 20px; border-radius: 5px; background-image: url('https://media.giphy.com/media/7VzgMsB6FLCilwS30v/giphy.gif'); background-size: cover; background-repeat: no-repeat;">
            <p style="font-size: 20px; font-weight: bold; margin-bottom: 10px; color: white;">Developer Details:</p>
            <p style="font-size: 18px; color: white;"><b>Name:</b> Somaraju Laxman Prasad</p>
            <p style="font-size: 18px; color: white;"><b>Study:</b> Masters in Data Analytics Level 9</p>
        """,
        unsafe_allow_html=True,
    )
    open_link_in_new_tab("https://www.datascienceportfol.io/laxmanprasad", "View More data Projects")
    open_link_in_new_tab("https://www.linkedin.com/in/laxman-prasad-somaraju-48a397229/", "View LinkedIn")
    st.markdown("</div>", unsafe_allow_html=True)


# Function to display progress information in a rectangle
def show_progress_info():
    st.markdown(
        """
        <div style="border: 1px solid #ccc; padding: 20px; border-radius: 5px; margin-top: 20px; background-image: url('https://media.giphy.com/media/Qvp6Z2fidQR34IcwQ5/giphy.gif'); background-size: cover; background-repeat: no-repeat;">
            <p style="font-size: 21px; font-weight: bold; margin-bottom: 10px; color: red;">Note:</p>
            <p style="font-size: 20px; font-weight: bold; color: white;">Please note that our app's success prediction for smartphones and laptops is based solely on historical data and machine learning algorithms. While we strive to provide valuable insights into the likelihood of success in the market, it's important to understand that the predictions do not take into account all marketing strategies, external influences, or real-time market dynamics. The app serves as an informative tool to aid in decision-making by analyzing past patterns, but actual market outcomes may be influenced by a wide range of factors beyond the scope of the model. Users should consider the predictions as one aspect of their overall strategy and complement them with comprehensive market research and marketing ideas to maximize their products' chances of success.</p>
            <p style="font-size: 20px; font-weight: bold; color: white;">you can check my GitHub repository for the code and app development progress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Set the page background color to black
st.markdown(
    """
    <style>
    body {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display developer details
show_developer_details()






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




# Button to predict the device success rate for phones
if st.button("Predict It - Phones", key="predict_phones"):
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
            st.warning("Product will be a failure in the current Market")
            # Display prediction result with a pie chart

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

# Button to predict the device success rate for laptops
if st.button("Predict It - Laptops", key="predict_laptops"):
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
        st.error("Invalid input. Please ensure all numeric values are greater than zero.")
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
            st.error("Contact Dev as there is an issue with the code")
        else:
            # Convert the data to numpy array and reshape it
            data = np.array(data).reshape(1, -1)

            # Make prediction using model_2
            prediction = model_2.predict(data)
            if prediction is True:
                st.success("Product should be a success in the current Market")
            else:
                st.warning("Product will be a failure in the current Market")


# Define the location of the images
image_locations = [
    "./phones/src/models/models_information/best_modelafterandbeforetuning.png",
    "./phones/src/models/models_information/confusion_matrix_roc_curve.png",
    "./phones/src/models/models_information/confusionmatrixbeforeaftertuning.png",
    "./phones/src/models/models_information/cross_validation_score.png",
    "./phones/src/models/models_information/cross_validation.png",
    "./phones/src/models/models_information/ensemblemethods.png",
    "./phones/src/models/models_information/feature_important.png",
    "./phones/src/models/models_information/hypertuning.png",
    "./phones/src/models/models_information/output.png",
    "./phones/src/models/models_information/rocbeforeaftertuning.png",
    "./phones/src/models/models_information/tetsingaccurcyofensemblemethods.png",
    "./phones/src/models/models_information/training_and_testing_confusion_matrix.png",
]

# Description of the visualizations
st.markdown("""
This section contains visualizations based on the modeling process for Phones. It includes ensemble methods, cross-validation, feature importance, best model before and after tuning, confusion matrix, ROC curve, accuracy of models for testing and training. If you want to view these visualizations, click the button below:
""")
            
# Button to display the images
if st.button("View Charts for the Phone Dataset After Modeling"):
    for image_path in image_locations:
        image = Image.open(image_path)
        st.image(image, caption=image_path, use_column_width=True)
    # Add a "Back" button to hide the images
    if st.button("Back"):
        pass  # This will not execute any code, effectively hiding the images when clicked on "Back"

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
if st.button("View Phone Dataset by downloading from the github and drop in here"):
    view_dataset()

# Link to view the dataset on GitHub
github_link = "https://github.com/laxmanprasadsomarajutus/myproject/blob/main/phones/data/processed-v2.csv"
if st.button("View Phone Dataset on GitHub"):
    st.markdown(f"Click [here]({github_link}) to view the dataset on GitHub.")


# Create a section for all graphs
st.subheader("Graphs for Analysis(Phones)")

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
# Button to display the information about the app and modeling
if st.button("About the App and Modeling"):
    st.markdown(
        """
        **About the App and Modeling:**

        Our app leverages the power of machine learning to predict the potential success of smartphones in the market. Using a trained Random Forest model, the app takes smartphone features as input and provides users with insights into the likelihood of success in the market.

        **Smartphone Success Prediction:**
        - **Input:** Users can input various smartphone attributes, including 'mrp', 'ram', 'inbuilt_storage', 'weight', 'battery_power', 'battery_type', 'form_factor', 'length', 'width', 'height', 'os_name', 'os_version', 'phone_warranty (months)', 'camera_count', 'cam_has_AI', 'cam_has_OIS', 'cam_has_Zoom', 'cam_has_HDR', 'cam_has_Macro', and 'cam_has_Portrait'.
        - **Prediction:** Our trained Random Forest model uses the provided features to make predictions on the potential success of the smartphone. The prediction result will indicate whether the smartphone is likely to be successful or not in the market.

        **Insights for Decision Making:**
        - The app offers valuable insights into the importance of different smartphone features in determining success. Users can identify key factors that positively or negatively impact the predicted success of their smartphones.
        - Please note that the predictions are based on historical patterns observed in the training data and are meant to be informative. Actual market outcomes may be influenced by various external factors beyond the scope of the model.

        **Modeling Approach:**
        - The Random Forest model used in the app was trained on a diverse and representative dataset, consisting of various smartphone features and their corresponding market outcomes.
        - Feature engineering techniques were employed to create relevant features like 'avg_rating' and 'percentage' success metrics, contributing to the accuracy of the model's predictions.
        - The selection of thresholds and weightages for the success metric was iteratively fine-tuned to achieve a balanced representation of smartphone success factors. While the initial values were randomly assigned, the refinement process involved trial and error and exploratory data analysis.

        By combining the power of machine learning and feature engineering, our app aims to assist users in making informed decisions when designing and marketing their smartphones, ultimately enhancing their chances of success in the competitive market.
        """
    )

if st.button("About the App and Modeling(laptop)"):    # Displaying the description for Laptop section
    st.markdown(
    """
    **About the App and Modeling (Laptop):**

    Our app goes beyond predicting smartphone success; it also extends its capabilities to classify the potential success of laptops in the market. Powered by advanced machine learning techniques, our app takes laptop features as input and provides users with valuable insights into the likelihood of success in the competitive laptop market.

    **Laptop Success Classification:**
    - **Input:** Users can input a range of laptop specifications, including 'mrp', 'os', 'hard_disk_type', 'ram_memory', 'processor_brand', 'processor_count', 'display_type', 'form_factor', 'no of 5 star', 'no of 4 star', 'no of 3 star', 'no of 2 star', 'no of 1 star', 'screen_res_w', 'screen_res_h', 'length', 'width', 'height', 'weight', 'hard_drive_size_value', 'hard_drive_size_unit', and 'battery_type'.
    - **Prediction:** Utilizing a carefully trained Random Forest model, the app analyzes the provided laptop features to predict its potential success in the market. The prediction result will indicate whether the laptop is likely to be successful or not.

    **Insights for Decision Making:**
    - Our app doesn't stop at just providing predictions. It offers valuable insights into the significance of different laptop features in determining success. Users can identify key factors that positively or negatively impact the predicted success of their laptops.
    - Please note that the predictions are based on historical patterns observed in the training data and serve as informative guidelines. The actual market performance of laptops may be influenced by a multitude of external factors beyond the scope of the model.

    **Modeling Approach:**
    - Similar to the smartphone success prediction, our Random Forest model for laptops was trained on a diverse and representative dataset. This dataset contains a wide array of laptop features and their corresponding market outcomes, ensuring the model's ability to generalize to various laptop models.
    - Feature engineering techniques were applied to extract relevant features, such as 'avg_rating' and 'percentage' success metrics, further contributing to the model's accuracy in predicting laptop success.
    - The determination of thresholds and weightages for the success metric followed an iterative fine-tuning process. The initial values were randomly assigned, but through meticulous trial and error and exploratory data analysis, we arrived at refined values for a well-balanced representation of laptop success factors.

    By leveraging the power of machine learning and feature engineering, our app aims to empower users in making informed decisions when designing, manufacturing, and marketing their laptops. We believe that these insights will enhance their chances of achieving success in the dynamic and competitive laptop market.
    """
)
# Display progress information
show_progress_info()
