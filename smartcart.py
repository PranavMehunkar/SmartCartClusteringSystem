import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="SmartCart Customer Segmentation", layout="wide")
st.title("🛒 SmartCart Customer Clustering System")

# ----------------------------------
# Load data
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("smartcart_customers.csv")

df = load_data()

# ----------------------------------
# Preprocessing & Feature Engineering
# ----------------------------------
@st.cache_data
def preprocess_data(df):
    df = df.copy()

    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (ref_date - df["Dt_Customer"]).dt.days

    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })

    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })

    drop_cols = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome",
        "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]

    return df.drop(columns=drop_cols)

df_cleaned = preprocess_data(df)

# ----------------------------------
# Encoding & Scaling
# ----------------------------------
@st.cache_data
def encode_scale(df):
    cat_cols = ["Education", "Living_With"]

    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(df[cat_cols])

    enc_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index
    )

    df_final = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)

    return X_scaled, df_final

X_scaled, df_final = encode_scale(df_cleaned)

# ----------------------------------
# PCA + Clustering
# ----------------------------------
@st.cache_resource
def run_clustering(X):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    sil_score = silhouette_score(X_pca, labels)

    return X_pca, labels, sil_score

X_pca, labels, sil_score = run_clustering(X_scaled)

df_final["Cluster"] = labels

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("Controls")
show_plot = st.sidebar.checkbox("Show 3D Cluster Plot")
show_summary = st.sidebar.checkbox("Show Cluster Summary")

# ----------------------------------
# Metrics
# ----------------------------------
st.subheader("📊 Model Performance")
st.metric("Silhouette Score", round(sil_score, 3))
st.metric("Number of Customers", df_final.shape[0])
st.metric("Clusters", 4)

# ----------------------------------
# 3D Plot
# ----------------------------------
if show_plot:
    st.subheader("🧠 3D PCA Cluster Visualization")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=labels
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    st.pyplot(fig)

# ----------------------------------
# Cluster Summary
# ----------------------------------
if show_summary:
    st.subheader("📋 Cluster Summary")
    summary = df_final.groupby("Cluster").mean()
    st.dataframe(summary)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("SmartCart Customer Segmentation using KMeans & PCA")
