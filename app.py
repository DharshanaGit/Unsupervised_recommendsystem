import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Flipkart Product Recommendation System", layout="wide")

# ===============================
# LOAD MODELS
# ===============================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
le_brand = joblib.load("brand_encoder.pkl")
le_category = joblib.load("category_encoder.pkl")
cluster_labels = joblib.load("cluster_labels.pkl")

# ===============================
# LOAD DATA FOR RECOMMENDATION
# ===============================
df = pd.read_csv(r"flipkart_com-ecommerce_sample.csv")

df['brand'].fillna('Unknown', inplace=True)
df.dropna(subset=['retail_price', 'discounted_price'], inplace=True)

df['rating'] = df['product_rating'].apply(
    lambda x: float(x) if str(x).replace('.','',1).isdigit() else 0
)

df['discount_pct'] = (
    (df['retail_price'] - df['discounted_price']) / df['retail_price']
) * 100

df['main_category'] = df['product_category_tree'].apply(
    lambda x: str(x).split(">>")[0].replace('[','').replace('"','').strip()
)

df['brand_encoded'] = le_brand.transform(df['brand'])
df['category_encoded'] = le_category.transform(df['main_category'])

# ===============================
# SIDEBAR – PROJECT DETAILS
# ===============================
st.sidebar.title("📌 Project Details")

st.sidebar.markdown("""
### 🔍 Algorithms Used
- K-Means Clustering  
- Hierarchical Clustering  
- DBSCAN  

### ✅ Selected Algorithm
**K-Means**  
Chosen for:
- Fast clustering
- Easy interpretation
- Scales well for products

### 🎯 Objective
Recommend similar products based on:
- Price
- Discount
- Rating
- Brand
- Category
""")
# Predict clusters for all products
features = df[['retail_price', 'discount_pct', 'rating', 'brand_encoded', 'category_encoded']]
features_scaled = scaler.transform(features)
df['cluster'] = kmeans.predict(features_scaled)

# Define cluster labels
cluster_labels = {
     0: "Best Value Products (High Rating & Good Discount)",
    1: "Low Engagement Products (Low Discount, Low Rating)",
    2: "Heavy Discount Products (Price-driven Buyers)",
    3: "Premium & Luxury Products",
    4: "Budget Products (Low Price, Low Engagement)",
    5: "High Discount – Low Rated Products",
    6: "Mid-Range Products (Balanced Features)",
    7: "Specialty Products (Niche Market)",
    8: "Trending Products",
    9: "Clearance Products"
}
# ===============================
# MAIN UI
# ===============================
st.title("🛍️ Flipkart Product Recommendation System")

st.markdown("### Enter your preferences")

col1, col2, col3 = st.columns(3)

with col1:
    budget = st.slider("💰 Budget (₹)", 500, 50000, 15000)

with col2:
    min_discount = st.slider("🔥 Minimum Discount (%)", 0, 80, 30)

with col3:
    min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 4.0)

brand = st.selectbox("🏷️ Preferred Brand", le_brand.classes_)
category = st.selectbox("📦 Product Category", le_category.classes_)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# PREDICT & RECOMMEND
# ===============================
if st.button("🎯 Predict & Recommend", use_container_width=True):

    # Encode inputs
    brand_enc = le_brand.transform([brand])[0]
    category_enc = le_category.transform([category])[0]

    user_features = np.array([[
        budget,
        min_discount,
        min_rating,
        brand_enc,
        category_enc
    ]])

    user_scaled = scaler.transform(user_features)

    cluster = kmeans.predict(user_scaled)[0]
    cluster_name = cluster_labels[cluster]

    st.success(f"### ✅ You belong to: **{cluster_name}**")

    # ===============================
    # RECOMMENDATION LOGIC
    # ===============================
    filtered = df[
        (df['cluster'] == cluster) &
        (df['retail_price'] <= budget) &
        (df['discount_pct'] >= min_discount) &
        (df['rating'] >= min_rating)
    ]

    if filtered.empty:
        st.warning("No exact matches found. Showing closest alternatives.")
        filtered = df[df['cluster'] == cluster]

    filtered['similarity'] = (
        abs(filtered['retail_price'] - budget) +
        abs(filtered['rating'] - min_rating) * 1000
    )

    recommendations = filtered.sort_values('similarity').head(5)

    st.markdown("### 🔥 Recommended Products")

    for _, row in recommendations.iterrows():
        st.write(f"""
        **🛒 {row['product_name'][:80]}**  
        💰 Price: ₹{row['retail_price']}  
        🔥 Discount: {row['discount_pct']:.1f}%  
        ⭐ Rating: {row['rating']}  
        🏷️ Brand: {row['brand']} | 📦 {row['main_category']}
        ---
        """)
    st.markdown("### Note: Recommendations are based on clustering similar products using K-Means algorithm.")