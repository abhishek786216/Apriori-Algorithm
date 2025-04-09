import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from io import BytesIO

# Page config
st.set_page_config(page_title="Market Basket Analyzer", page_icon="🧺", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding: 1rem 2rem; }
    .metric-box { background-color: #1e1e2f; padding: 1rem; border-radius: 12px; color: white; text-align: center; }
    .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 20px; border-radius: 8px; }
    .stButton>button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

st.title("🧺 Market Basket Analyzer")
st.caption("Discover hidden patterns in transactions using Apriori & FP-Growth")

# Sidebar - Dataset selection
st.sidebar.header("📂 Data Options")

default_data = {
    "Groceries 🛒": [["Milk", "Bread", "Butter"], ["Bread", "Butter", "Cheese"],
                     ["Milk", "Bread", "Cheese"], ["Bread", "Butter"],
                     ["Milk", "Butter", "Cheese"], ["Milk", "Bread", "Butter", "Cheese"]],
    "Fruits 🍎": [["Apple", "Banana", "Grapes"], ["Banana", "Grapes", "Mango"],
                 ["Apple", "Banana", "Mango"], ["Banana", "Grapes"],
                 ["Apple", "Grapes", "Mango"], ["Apple", "Banana", "Grapes", "Mango"]],
    "Electronics 💻": [["Laptop", "Mouse", "Keyboard"], ["Keyboard", "Monitor", "Mouse"],
                      ["Laptop", "Keyboard", "Monitor"], ["Keyboard", "Mouse"],
                      ["Laptop", "Monitor", "Mouse"], ["Laptop", "Mouse", "Keyboard", "Monitor"]]
}

data_mode = st.sidebar.radio("Choose Data Source:", ["Use Default Data", "Upload Your Own Data"])
selected_datasets = []

if data_mode == "Use Default Data":
    for name in default_data:
        if st.sidebar.checkbox(name, value=(name == "Groceries 🛒")):
            selected_datasets.append(pd.DataFrame(default_data[name]))

else:
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file, header=None)
            selected_datasets.append(df)
            st.sidebar.success(f"✅ {file.name} uploaded!")

# Algorithm choice
algo = st.sidebar.selectbox("🧠 Select Algorithm", ["Apriori", "FP-Growth"])
support = st.sidebar.slider("Support (min):", 0.01, 1.0, 0.04, 0.01)
confidence = st.sidebar.slider("Confidence (min):", 0.01, 1.0, 0.06, 0.01)

st.markdown("### ⚙️ Algorithm Settings")
st.write(f"- **Algorithm:** {algo}  \n- **Support ≥** {support:.2f}  \n- **Confidence ≥** {confidence:.2f}")

# Run algorithm
if st.button("🚀 Run Analysis"):
    if selected_datasets:
        combined_data = pd.concat(selected_datasets, ignore_index=True)
        records = [[str(combined_data.values[i, j]) for j in range(combined_data.shape[1]) if pd.notna(combined_data.values[i, j])] for i in range(len(combined_data))]

        encoder = TransactionEncoder()
        encoded_data = encoder.fit_transform(records)
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

        if algo == "Apriori":
            freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
        else:
            freq_items = fpgrowth(df_encoded, min_support=support, use_colnames=True)

        rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)

        if not rules.empty:
            st.success("🎉 Rules Generated Successfully!")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<div class="metric-box"><h3>📦 {len(freq_items)}</h3><p>Frequent Itemsets</p></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-box"><h3>🔗 {len(rules)}</h3><p>Rules Found</p></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-box"><h3>⚡ {rules["lift"].max():.2f}</h3><p>Max Lift</p></div>', unsafe_allow_html=True)

            # Rule Viewer
            st.markdown("### 📜 Association Rules")
            for _, row in rules.iterrows():
                st.write(f"**{set(row['antecedents'])} → {set(row['consequents'])}**")
                st.caption(f"Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
                st.divider()

            # Download option
            csv = rules.to_csv(index=False).encode()
            st.download_button("📥 Download Rules CSV", data=csv, file_name="rules.csv", mime="text/csv")

            # Graph visualization (Plotly)
            st.markdown("### 🔗 Relationship Graph")
            G = nx.DiGraph()
            for _, row in rules.iterrows():
                for antecedent in row["antecedents"]:
                    for consequent in row["consequents"]:
                        G.add_edge(antecedent, consequent, weight=row["lift"])

            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = np.random.rand(2)
                x1, y1 = np.random.rand(2)
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#FF5733'),
                                    hoverinfo='none', mode='lines')

            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = np.random.rand(2)
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                    hoverinfo='text', text=node_text,
                                    marker=dict(color='#00BFFF', size=20, line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(title='Association Graph',
                                             titlefont_size=16,
                                             showlegend=False,
                                             hovermode='closest',
                                             margin=dict(b=20,l=5,r=5,t=40)))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ No rules found for selected parameters.")
    else:
        st.error("Please select or upload at least one dataset.")
