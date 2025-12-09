# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("furniture_price_model.pkl")

# Helper functions
def size_feature(title):
    import re
    title_lower = str(title).lower()
    size_match = re.search(r'(\d+(\.\d+)?)\s*(inch|ft|cm|m)', title_lower)
    if size_match:
        value = float(size_match.group(1))
        unit = size_match.group(3)
        if unit == 'ft': return value * 12
        elif unit == 'cm': return value / 2.54
        elif unit == 'm': return value * 39.37
        return value
    return np.nan

def extract_material(title):
    title_lower = str(title).lower()
    materials = ['wood', 'metal', 'fabric', 'leather', 'plastic', 'glass', 'velvet', 'boucle']
    for mat in materials:
        if mat in title_lower:
            return mat
    return 'other'

def extract_color(title):
    title_lower = str(title).lower()
    colors = ['white', 'black', 'grey', 'gray', 'brown', 'blue', 'green', 'red', 'pink', 'yellow']
    for col in colors:
        if col in title_lower:
            return col
    return 'other'

# Session state for prediction history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# App UI
st.set_page_config(page_title="🛋️ Furniture Price Predictor", layout="centered")
st.title("🛋️ Furniture Price Prediction App")
st.write("Enter furniture details to predict the price:")

# Inputs
product_title = st.text_input("Product Title", "")
furniture_type = st.selectbox("Furniture Type", ["Chair", "Table", "Dresser", "Sofa", "Bed", "Other"])
sold = st.number_input("Units Sold", min_value=0, value=0, step=1)
material_input = st.selectbox("Material", ["Auto Detect", "Wood", "Metal", "Fabric", "Leather", "Plastic", "Glass", "Velvet", "Boucle", "Other"])
color_input = st.selectbox("Color", ["Auto Detect", "White", "Black", "Grey", "Brown", "Blue", "Green", "Red", "Pink", "Yellow", "Other"])
original_price = st.number_input("Original Price ($)", min_value=0.0, value=0.0)
discount_pct = st.number_input("Discount Percentage (0-100)", min_value=0.0, max_value=100.0, value=0.0) / 100
delivery_fee = st.number_input("Delivery Fee ($)", min_value=0.0, value=0.0)

# Feature extraction if "Auto Detect" selected
material = extract_material(product_title) if material_input == "Auto Detect" else material_input.lower()
color = extract_color(product_title) if color_input == "Auto Detect" else color_input.lower()
size_feat = size_feature(product_title)

# Prepare input for model
input_df = pd.DataFrame([{
    "productTitle": product_title,
    "sold": sold,
    "sizeFeat": size_feat,
    "material": material,
    "color": color,
    "discount_pct": discount_pct
}])

# Predict
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0] + delivery_fee
        st.success(f"📌 Predicted Price: ${prediction:.2f}")
        
        # Save to session history
        st.session_state['history'].append({
            "Title": product_title,
            "Type": furniture_type,
            "Sold": sold,
            "Material": material,
            "Color": color,
            "Discount": discount_pct*100,
            "Delivery": delivery_fee,
            "Predicted Price": round(prediction,2)
        })
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Show prediction history
if st.session_state['history']:
    st.subheader("Prediction History")
    st.table(pd.DataFrame(st.session_state['history']))
