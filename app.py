import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Set page config
st.set_page_config(page_title="Apriori Algorithm", page_icon="🛍️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .css-1aumxhk { background-color: #121212; }
    .st-bb { background-color: #212121; border-radius: 10px; padding: 20px; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 10px; font-size: 16px; padding: 10px; }
    .stButton>button:hover { background-color: #e63946; }
    .stSlider>div { width: 80%; }
    .stAlert { background-color: #2a9d8f; color: white; padding: 15px; border-radius: 10px; }
    .dataset-card { background: linear-gradient(135deg, #1f1c2c, #928dab); padding: 15px; margin: 10px 0;
                    border-radius: 10px; color: white; text-align: center; font-size: 16px; font-weight: bold;
                    transition: 0.3s; }
    .dataset-card:hover { transform: scale(1.05); box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.3); }
    </style>
""", unsafe_allow_html=True)

# Sidebar for dataset selection
st.sidebar.header("📂 Select Dataset")

# Predefined datasets
default_data1 = [["Milk", "Bread", "Butter"], ["Bread", "Butter", "Cheese"],
                ["Milk", "Bread", "Cheese"], ["Bread", "Butter"],
                ["Milk", "Butter", "Cheese"], ["Milk", "Bread", "Butter", "Cheese"]]

default_data2 = [["Apple", "Banana", "Grapes"], ["Banana", "Grapes", "Mango"],
                ["Apple", "Banana", "Mango"], ["Banana", "Grapes"],
                ["Apple", "Grapes", "Mango"], ["Apple", "Banana", "Grapes", "Mango"]]

default_data3 = [["Laptop", "Mouse", "Keyboard"], ["Keyboard", "Monitor", "Mouse"],
                ["Laptop", "Keyboard", "Monitor"], ["Keyboard", "Mouse"],
                ["Laptop", "Monitor", "Mouse"], ["Laptop", "Mouse", "Keyboard", "Monitor"]]

# Convert datasets into DataFrames
default_df1 = pd.DataFrame(default_data1)
default_df2 = pd.DataFrame(default_data2)
default_df3 = pd.DataFrame(default_data3)

# Dataset selection mode
data_mode = st.sidebar.radio("Choose Data Source:", ["Use Default Data", "Upload Your Own Data"], index=0)
selected_datasets = []

if data_mode == "Use Default Data":
    st.sidebar.markdown("### 📁 Select Default Datasets:")
    use_df1 = st.sidebar.checkbox("🛒 Groceries", value=True)
    use_df2 = st.sidebar.checkbox("🍎 Fruits", value=False)
    use_df3 = st.sidebar.checkbox("💻 Electronics", value=False)
    if use_df1: selected_datasets.append(default_df1)
    if use_df2: selected_datasets.append(default_df2)
    if use_df3: selected_datasets.append(default_df3)

elif data_mode == "Upload Your Own Data":
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file, header=None)
            selected_datasets.append(df)
            st.sidebar.write(f"✅ {file.name} uploaded!")

# Support and Confidence sliders
st.markdown("### 📊 Set Apriori Parameters")
support = st.slider("Select Support", 0.01, 1.0, 0.04, 0.01)
confidence = st.slider("Select Confidence", 0.01, 1.0, 0.06, 0.01)

# Run Apriori Algorithm
if st.button("🚀 Run Apriori Algorithm"):
    if selected_datasets:
        combined_data = pd.concat(selected_datasets, ignore_index=True)
        records = [[str(combined_data.values[i, j]) for j in range(combined_data.shape[1]) if pd.notna(combined_data.values[i, j])] for i in range(len(combined_data))]
        encode = TransactionEncoder()
        data_encoded = encode.fit_transform(records)
        data_df = pd.DataFrame(data_encoded, columns=encode.columns_)
        frequent_data = apriori(data_df, min_support=support, use_colnames=True)
        rules = association_rules(frequent_data, min_threshold=confidence)
        
        if not rules.empty:
            st.success("✅ Association Rules Generated Successfully!")
            for index, row in rules.iterrows():
                st.write(f"**Rule:** {list(row['antecedents'])} → {list(row['consequents'])}")
                st.write(f"**Support:** {row['support']:.4f} | **Confidence:** {row['confidence']:.4f} | **Lift:** {row['lift']:.4f}")
                st.write("---")
            
            # Visualization
            st.markdown("### 📊 Visualization of Relationships")
            G = nx.DiGraph()
            for index, row in rules.iterrows():
                antecedents = list(row["antecedents"])
                consequents = list(row["consequents"])
                for antecedent in antecedents:
                    for consequent in consequents:
                        G.add_edge(antecedent, consequent, weight=row['lift'])
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(10, 8))
            nx.draw_networkx_nodes(G, pos, node_size=5000, node_color="#90CAF9", alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="#FF7043")
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            st.pyplot(plt)
        else:
            st.warning("⚠️ No association rules found based on the provided support and confidence values.")
    else:
        st.warning("⚠️ Please select at least one dataset to proceed.")
