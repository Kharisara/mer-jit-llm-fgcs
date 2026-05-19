import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="JITAI Policy Dashboard", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = "paper_outputs"

POLICY_FIRST = os.path.join(OUTPUT_DIR, "policy_first_outputs.csv")
PROXY = os.path.join(OUTPUT_DIR, "proxy_outputs.csv")
E2E = os.path.join(OUTPUT_DIR, "e2e_outputs.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_policy = load_csv(POLICY_FIRST)
df_proxy = load_csv(PROXY)
df_e2e = load_csv(E2E)

# -----------------------------
# HEADER
# -----------------------------
st.title("Policy-First JITAI Dashboard")
st.markdown("Evaluation of controllability, safety, and intervention behavior")

# -----------------------------
# ARCHITECTURE
# -----------------------------
st.header("System Architecture")

st.markdown("""
**Policy-First Decision Pipeline**

User Input → Policy → Decision (Act / Silence) → LLM → Output

- Policy controls intervention
- LLM only activates when allowed
- Enables safety + auditability
""")

# -----------------------------
# METRIC FUNCTION
# -----------------------------
def compute_metrics(df):
    total = len(df)
    interventions = int(df["action"].sum())
    silence = total - interventions

    unauthorized = df[
        (df["action"] == 0) & (df["reply_json"].notna())
    ]

    return {
        "total": total,
        "interventions": interventions,
        "silence": silence,
        "rate": interventions / total if total else 0,
        "unauthorized": len(unauthorized)
    }

# -----------------------------
# SUMMARY METRICS
# -----------------------------
st.header("Key Metrics")

col1, col2, col3 = st.columns(3)

if df_policy is not None:
    m = compute_metrics(df_policy)
    col1.metric("Policy Intervention Rate", f"{m['rate']:.2f}")
    col1.metric("Policy Unauthorized", m["unauthorized"])

if df_proxy is not None:
    m = compute_metrics(df_proxy)
    col2.metric("Proxy Intervention Rate", f"{m['rate']:.2f}")
    col2.metric("Proxy Unauthorized", m["unauthorized"])

if df_e2e is not None:
    total = len(df_e2e)
    interventions = int(df_e2e["action"].sum())
    rate = interventions / total if total else 0

    col3.metric("E2E Response Rate", f"{rate:.2f}")
    col3.metric("Silence Path", "None")

# -----------------------------
# BAR CHART COMPARISON
# -----------------------------
st.header("Intervention Comparison")

labels = []
rates = []

if df_policy is not None:
    labels.append("Policy")
    rates.append(compute_metrics(df_policy)["rate"])

if df_proxy is not None:
    labels.append("Proxy")
    rates.append(compute_metrics(df_proxy)["rate"])

if df_e2e is not None:
    total = len(df_e2e)
    interventions = int(df_e2e["action"].sum())
    rate = interventions / total if total else 0

    labels.append("E2E")
    rates.append(rate)

fig, ax = plt.subplots()
ax.bar(labels, rates)
ax.set_ylabel("Intervention Rate")
ax.set_title("Policy vs Proxy vs E2E")

st.pyplot(fig)

# -----------------------------
# ACTION DISTRIBUTION
# -----------------------------
st.header("Action Distribution")

def plot_action_distribution(df, title):
    counts = df["action"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Action (0 = Silence, 1 = Intervention)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

col1, col2 = st.columns(2)

if df_policy is not None:
    with col1:
        plot_action_distribution(df_policy, "Policy Actions")

if df_proxy is not None:
    with col2:
        plot_action_distribution(df_proxy, "Proxy Actions")

# -----------------------------
# SAFETY ANALYSIS
# -----------------------------
st.header("Safety Analysis")

if df_policy is not None:
    m = compute_metrics(df_policy)
    st.write("Policy Unauthorized Responses:", m["unauthorized"])

if df_proxy is not None:
    m = compute_metrics(df_proxy)
    st.write("Proxy Unauthorized Responses:", m["unauthorized"])

st.markdown("""
**Interpretation:**
- Unauthorized = response generated when policy said silence
- Policy-first should minimize this → demonstrates safety control
""")

# -----------------------------
# RAW DATA (for viva questions)
# -----------------------------
st.header("Raw Data Preview")

tab1, tab2, tab3 = st.tabs(["Policy", "Proxy", "E2E"])

with tab1:
    if df_policy is not None:
        st.dataframe(df_policy.head(50))

with tab2:
    if df_proxy is not None:
        st.dataframe(df_proxy.head(50))

with tab3:
    if df_e2e is not None:
        st.dataframe(df_e2e.head(50))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built for FYP Viva Demonstration")