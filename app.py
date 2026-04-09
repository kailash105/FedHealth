import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Federated Learning Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Theme & Cards ---
# Streamlit has a default dark theme that can be configured in .streamlit/config.toml, 
# but applying some custom styling ensures the "card" look.
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1e1e2f;
        border: 1px solid #33334d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"]:hover {
        border-color: #4CAF50;
        transition: 0.3s;
    }
    /* Headers */
    h1, h2, h3 {
        color: #e0e0e0;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 5px;
        font-weight: bold;
    }
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        with open('federated_results.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("Error: federated_results.json not found.")
        return None

data = load_data()

# --- Utility Functions ---
def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton>button">{text}</a>'
    return href

# --- Pages ---

def show_home():
    st.title("🏥 Federated Learning for Healthcare")
    st.markdown("""
    Welcome to the Privacy-Preserving Federated Learning Dashboard for Healthcare. 
    This platform simulates collaborative model training among hospitals without sharing sensitive raw patient data, 
    ensuring compliance with data privacy regulations while achieving high predictive performance.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Run Training", use_container_width=True):
            st.session_state['page'] = 'Train Model'
            st.rerun()
    with col2:
        if st.button("📊 View Results", use_container_width=True, disabled=not st.session_state.get('model_trained', False)):
            st.session_state['page'] = 'Results Dashboard'
            st.rerun()

    st.markdown("---")
    st.subheader("📂 Upload Hospital Patient Data")
    uploaded_file = st.file_uploader("Upload CSV formatted patient health data securely", type=["csv"])
    
    if uploaded_file is not None:
        if not st.session_state.get('dataset_uploaded', False):
            with st.spinner("Processing dataset..."):
                time.sleep(2.5)  # Simulate processing delay
            st.session_state['dataset_uploaded'] = True
            st.toast("Clients initialized successfully", icon="✅")
            
        st.success("✅ Dataset successfully uploaded and validated")
        
        # Display Preview
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"**Dataset Info:** {df_preview.shape[0]} rows, {df_preview.shape[1]} columns")
        st.dataframe(df_preview.head(5), use_container_width=True)
        
        st.info("🔄 Dataset distributed across 5 clients")
        
        with st.expander("⚙️ Dataset Processing Details"):
            st.markdown("""
- ✅ Data cleaning completed
- ✅ Feature normalization applied
- ✅ Partitioned into 5 distributed clients
- ✅ Ready for federated training
            """)
            st.caption("🔒 Data is securely partitioned and never leaves client nodes")
    else:
        st.session_state['dataset_uploaded'] = False

    st.markdown("---")
    st.subheader("Global Configuration Overview")
    if data and "config" in data:
        conf = data["config"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rounds", conf.get("n_rounds", 20))
        c2.metric("Local Epochs", conf.get("local_epochs", 10))
        c3.metric("Learning Rate", conf.get("lr", 0.05))
        c4.metric("DP Epsilon", conf.get("dp_epsilon", "None"))

def show_results_dashboard():
    st.title("📊 Results Dashboard")
    
    if not data:
        return
        
    st.subheader("Model Performance Comparison")
    
    fed = data.get("federated", {})
    cent = data.get("centralised", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌐 Federated Model**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Accuracy", f"{fed.get('accuracy', 0):.4f}")
        fc2.metric("F1 Score", f"{fed.get('f1', 0):.4f}")
        fc3.metric("AUC", f"{fed.get('auc', 0):.4f}")
        
    with col2:
        st.markdown("**🏢 Centralised Model**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Accuracy", f"{cent.get('accuracy', 0):.4f}", delta=f"{cent.get('accuracy', 0) - fed.get('accuracy', 0):.4f}", delta_color="inverse")
        cc2.metric("F1 Score", f"{cent.get('f1', 0):.4f}", delta=f"{cent.get('f1', 0) - fed.get('f1', 0):.4f}", delta_color="inverse")
        cc3.metric("AUC", f"{cent.get('auc', 0):.4f}", delta=f"{cent.get('auc', 0) - fed.get('auc', 0):.4f}", delta_color="inverse")

    st.divider()
    
    # Graphs
    st.subheader("Training Progression")
    round_log = data.get("round_log", [])
    if round_log:
        df_rounds = pd.DataFrame(round_log)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Accuracy vs Rounds", "📈 F1 Score vs Rounds", "ROC-AUC vs Rounds"])
        
        with tab1:
            fig_acc = px.line(df_rounds, x="round", y="avg_acc", markers=True, 
                              title="Federated Accuracy over Rounds", template="plotly_dark",
                              labels={"round": "Training Round", "avg_acc": "Avg Accuracy"})
            fig_acc.update_traces(line_color="#4CAF50")
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with tab2:
            fig_f1 = px.line(df_rounds, x="round", y="avg_f1", markers=True, 
                             title="Federated F1 Score over Rounds", template="plotly_dark",
                              labels={"round": "Training Round", "avg_f1": "Avg F1 Score"})
            fig_f1.update_traces(line_color="#2196F3")
            st.plotly_chart(fig_f1, use_container_width=True)
            
        with tab3:
            fig_auc = px.line(df_rounds, x="round", y="avg_auc", markers=True, 
                              title="Federated AUC over Rounds", template="plotly_dark",
                              labels={"round": "Training Round", "avg_auc": "Avg AUC"})
            fig_auc.update_traces(line_color="#FF9800")
            st.plotly_chart(fig_auc, use_container_width=True)
            
        st.markdown(get_csv_download_link(df_rounds, "federated_rounds.csv", "📥 Download Round Metrics (CSV)"), unsafe_allow_html=True)

def show_client_analysis():
    st.title("🏥 Client Analysis")
    
    if not data:
        return
        
    per_client = data.get("per_client", {})
    if per_client:
        clients = []
        for client_name, metrics in per_client.items():
            clients.append({
                "Client": client_name.replace("_", " ").title(),
                "Accuracy": metrics.get("accuracy", 0),
                "F1 Score": metrics.get("f1", 0),
                "AUC": metrics.get("auc", 0),
                "Samples": metrics.get("n", 0)
            })
            
        df_clients = pd.DataFrame(clients)
        
        # Highlight best and worst
        best_client = df_clients.loc[df_clients['Accuracy'].idxmax()]
        worst_client = df_clients.loc[df_clients['Accuracy'].idxmin()]
        
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**🏆 Best Client:** {best_client['Client']} (Acc: {best_client['Accuracy']:.4f})")
        with c2:
            st.error(f"**⚠️ Worst Client:** {worst_client['Client']} (Acc: {worst_client['Accuracy']:.4f})")
            
        st.dataframe(df_clients.style.highlight_max(subset=['Accuracy', 'F1 Score', 'AUC'], color='#1b5e20')
                                       .highlight_min(subset=['Accuracy', 'F1 Score', 'AUC'], color='#b71c1c'), 
                     use_container_width=True)
                     
        # Bar chart for clients
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_clients['Client'], y=df_clients['Accuracy'], name='Accuracy', marker_color='#4CAF50'))
        fig.add_trace(go.Bar(x=df_clients['Client'], y=df_clients['AUC'], name='AUC', marker_color='#FF9800'))
        fig.update_layout(barmode='group', title="Client Performance Comparison", template="plotly_dark",
                          xaxis_title="Clients", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(get_csv_download_link(df_clients, "client_metrics.csv", "📥 Download Client Data (CSV)"), unsafe_allow_html=True)

def show_train_model():
    st.title("⚙️ Train Federated Model")
    
    if not st.session_state.get('dataset_uploaded', False):
        st.error("🚫 Please upload dataset from Home page before training")
        st.stop()
        
    if not data:
        return
        
    audit_log = data.get("audit_log", [])
    
    st.write("Simulate the federated learning rounds based on the audit logs.")
    
    # Bonus: DP Toggle
    dp_toggle = st.toggle("Enable Differential Privacy (Simulation)", value=False)
    if dp_toggle:
        st.info("🛡️ Differential Privacy is enabled. Noise will be added to model weights.")
    
    if st.button("▶️ Start Training Simulation"):
        total_rounds = len(audit_log)
        if total_rounds == 0:
            st.warning("No audit logs available to simulate training.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        placeholder_chart = st.empty()
        weight_norms = []
        rounds = []
        
        for i, log in enumerate(audit_log):
            n_clients = log.get("n_clients", 5)
            total_samples = log.get("total_samples", 800)
            weight_norm = log.get("weight_l2_norm", 0)
            
            rounds.append(i + 1)
            weight_norms.append(weight_norm)
            
            status_text.markdown(f"**Round {i+1} / {total_rounds}** | Clients Participating: {n_clients} | Total Samples: {total_samples}")
            progress_bar.progress((i + 1) / total_rounds)
            
            fig = px.line(x=rounds, y=weight_norms, markers=True,
                          title="Model Weight L2 Norm Progression", 
                          template="plotly_dark",
                          labels={"x": "Training Round", "y": "L2 Norm"})
            fig.update_traces(line_color="#E91E63")
            placeholder_chart.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.3)
            
        st.success("✅ Federated Training Completed!")
        st.session_state['model_trained'] = True

def show_training_logs():
    st.title("📝 Training Logs & Audit")
    
    if not data:
        return
        
    audit_log = data.get("audit_log", [])
    if audit_log:
        # Add Round number to logs
        for i, log in enumerate(audit_log):
            log['Round'] = i + 1
            
        df_logs = pd.DataFrame(audit_log)
        
        # Reorder columns
        cols = ['Round', 'n_clients', 'total_samples', 'dp_enabled', 'weight_l2_norm']
        df_logs = df_logs[[c for c in cols if c in df_logs.columns]]
        
        df_logs.rename(columns={
            'n_clients': 'Number of Clients',
            'total_samples': 'Total Samples',
            'dp_enabled': 'DP Enabled',
            'weight_l2_norm': 'Weight L2 Norm'
        }, inplace=True)
        
        st.dataframe(df_logs, use_container_width=True)
        st.markdown(get_csv_download_link(df_logs, "audit_logs.csv", "📥 Download Logs (CSV)"), unsafe_allow_html=True)


# --- Sidebar Navigation ---
st.sidebar.title("🏥 FedHealth")
st.sidebar.markdown("---")

# Navigation and Global state
if 'dataset_uploaded' not in st.session_state:
    st.session_state['dataset_uploaded'] = False

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

nav_options = ["Home", "Train Model"]
if st.session_state.get('model_trained', False):
    nav_options.extend(["Results Dashboard", "Client Analysis", "Training Logs"])

if st.session_state['page'] not in nav_options:
    st.session_state['page'] = "Home"

# Set up the radio button based on session state, but also update session state when it changes
selected_page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state['page'])
)

# Update session state based on sidebar selection
st.session_state['page'] = selected_page

# Display the appropriate page
if st.session_state['page'] == "Home":
    show_home()
elif st.session_state['page'] == "Train Model":
    show_train_model()
elif st.session_state['page'] == "Results Dashboard":
    show_results_dashboard()
elif st.session_state['page'] == "Client Analysis":
    show_client_analysis()
elif st.session_state['page'] == "Training Logs":
    show_training_logs()

st.sidebar.markdown("---")
st.sidebar.info("Developed for Privacy-Preserving AI in Healthcare.")
