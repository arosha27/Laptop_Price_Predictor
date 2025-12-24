# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # 1. Load the model
# # Ensure 'best_laptop_price_model.pkl' is in the same folder as this script
# pipe = pickle.load(open('TrainedModels/best_laptop_price_model.pkl', 'rb'))

# st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# st.title("💻 Laptop Price Predictor")
# st.write("Provide the specifications below to get the estimated price.")

# # --- INPUT FIELDS ---
# col1, col2 = st.columns(2)

# with col1:
#     company = st.selectbox('Brand', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Toshiba', 'Other'])
#     type_name = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
#     ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
#     weight = st.number_input('Weight of the Laptop (kg)', value=2.0)

# with col2:
#     touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
#     ips = st.selectbox('IPS Panel', ['No', 'Yes'])
#     screen_size = st.number_input('Screen Size (Inches)', value=15.6)
#     resolution = st.selectbox('Screen Resolution', [
#         '1920x1080', '1366x768', '1600x900', '3840x2160', 
#         '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
#     ])

# st.divider()

# col3, col4 = st.columns(2)

# with col3:
#     cpu = st.selectbox('CPU Brand', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor'])
#     hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
#     ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# with col4:
#     gpu = st.selectbox('GPU Brand', ['Intel', 'Nvidia', 'AMD'])
#     os_sys = st.selectbox('Operating System', ['Windows', 'Mac', 'Others/No OS/Linux'])

# # --- PREDICTION LOGIC ---
# if st.button('💰 Predict Price'):
#     # 1. Calculate PPI (Required feature for your model)
#     X_res = int(resolution.split('x')[0])
#     Y_res = int(resolution.split('x')[1])
#     ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

#     # 2. Prepare the Input DataFrame
#     # Note: Column names match your training set EXACTLY (TouchScreen, IPS, ppi)
#     query_df = pd.DataFrame({
#         'Company': [company],
#         'TypeName': [type_name],
#         'Ram': [ram],
#         'Weight': [weight],
#         'TouchScreen': [1 if touchscreen == 'Yes' else 0],
#         'IPS': [1 if ips == 'Yes' else 0],
#         'Cpu Brand': [cpu],
#         'HDD': [hdd],
#         'SSD': [ssd],
#         'Hybrid': [0],
#         'Flash_Storage': [0],
#         'Gpu Brand': [gpu],
#         'os': [os_sys],
#         'ppi': [ppi]
#     })

#     # 3. Predict and Inverse Log Transformation
#     # Since training used np.log(Price), we use np.exp() to get the real value
#     predicted_log_price = pipe.predict(query_df)
#     final_price = np.exp(predicted_log_price[0])

#     # 4. Display Result in RS
#     st.success(f"### The estimated price is RS {int(final_price):,}")
    
 
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the model (The Pipeline)
# Ensure this file exists in your project folder
pipe = pickle.load(open('TrainedModels/best_laptop_price_model.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.title("💻 Laptop Price Predictor & Insights")

# --- SIDEBAR FOR INSIGHTS ---
st.sidebar.header("Dashboard Options")
show_analysis = st.sidebar.checkbox("Show Feature Importance Graph")

# --- MAIN PREDICTION UI ---
st.subheader("Estimate Laptop Price")
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Toshiba', 'Other'])
    type_name = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight (kg)', value=2.0)

with col2:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Panel', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size (Inches)', value=15.6)
    resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

st.divider()

col3, col4 = st.columns(2)
with col3:
    cpu = st.selectbox('CPU Brand', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor'])
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
with col4:
    gpu = st.selectbox('GPU Brand', ['Intel', 'Nvidia', 'AMD'])
    os_sys = st.selectbox('OS', ['Windows', 'Mac', 'Others/No OS/Linux'])

if st.button('💰 Predict Price'):
    # Feature Engineering
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query_df = pd.DataFrame({
        'Company': [company], 'TypeName': [type_name], 'Ram': [ram], 'Weight': [weight],
        'TouchScreen': [1 if touchscreen == 'Yes' else 0], 'IPS': [1 if ips == 'Yes' else 0],
        'Cpu Brand': [cpu], 'HDD': [hdd], 'SSD': [ssd], 'Hybrid': [0],
        'Flash_Storage': [0], 'Gpu Brand': [gpu], 'os': [os_sys], 'ppi': [ppi]
    })

    result = pipe.predict(query_df)
    st.success(f"### The estimated price is RS {int(np.exp(result[0])):,}")

# --- FEATURE IMPORTANCE SECTION ---
if show_analysis:
    st.divider()
    st.subheader("🔍 Which factors contribute most to the price?")
    
    try:
        # Get the model and the transformer from the pipeline
        step1 = pipe.named_steps['step1']
        model = pipe.named_steps['step2']
        
        # Get feature names from the OneHotEncoder inside the ColumnTransformer
        ohe = step1.named_transformers_['col_tnf']
        
        # Get original column names to identify which are categorical
        # If your training code used specific column indices, adjust accordingly
        cat_cols = ['Company', 'TypeName', 'Cpu Brand', 'Gpu Brand', 'os']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        
        # Remaining columns (numerical)
        num_cols = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage', 'ppi']
        
        all_feature_names = cat_feature_names + num_cols
        importances = model.feature_importances_

        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plotting with Matplotlib/Seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='magma', ax=ax)
        plt.title('Top 10 Price Drivers')
        
        # Display in Streamlit
        st.pyplot(fig)
        st.info("The chart shows that CPU, RAM, and SSD typically have the highest impact on the price.")
        
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        st.warning("Make sure the pipeline was saved with the same column names used in the app.")