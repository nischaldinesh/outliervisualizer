import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
import io
from PIL import Image
from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

st.set_page_config(page_title="Outlier Dashboard", layout="wide")

# --- Styling ---
st.markdown("""
<h1 style='text-align: center; color: black; font-size: 32px; margin:1rem 0 1rem 0;'>
    Outlier Visualizer: A Dashboard for Comparative Analysis of Outlier Detection Algorithms
</h1>
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding: 1rem !important; }
    .highlight-header {
        background-color: #3b82f6;
        color: white;
        padding: 8px 10px;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 4px;
        text-align: center;
    }
    .algo-group {
        background-color: #717D7E;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .algo-group strong { color: white; font-size: 16px; }
    .stButton>button {
        height: 2.8rem; width: 100%;
        font-weight: 500; background-color: #3b82f6; color: white;
        border-radius: 0 !important;
    }
    .stButton>button[key="continue_btn"] {
        border-radius: 12px !important;
        margin-top: 20px !important;
        width: 40% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
            
    .small-header {
            padding" 4px 6px !important;
            font-size: 14px !important;
            margin-bottom: 4px !important;}
</style>
""", unsafe_allow_html=True)


# --- Caching ---
@st.cache_data(show_spinner=False)
def load_cached_dataset(name):
    df = load_dataset(name)
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    return SimpleImputer().fit_transform(df_numeric)

@st.cache_data(show_spinner=False)
def get_pca_coords(X):
    X2 = PCA(n_components=2).fit_transform(X)
    X2 = X2 / np.max(np.abs(X2)) + np.random.normal(0, 0.015, size=X2.shape)

    xpad = np.ptp(X2[:, 0]) * 0.1
    ypad = np.ptp(X2[:, 1]) * 0.1

    x_min, x_max = X2[:, 0].min() - xpad, X2[:, 0].max() + xpad
    y_min, y_max = X2[:, 1].min() - ypad, X2[:, 1].max() + ypad

    return X2, (x_min, x_max), (y_min, y_max)

@st.cache_data(show_spinner=False)
def get_scores_cached(algo, X):
    return run_algorithm(algo, X)


# --- Session state defaults ---
for key, default in [
    ("dataset", None), ("algorithms", {}), ("heatmap_type", None),
    ("colormap", None), ("confirmed", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Category setup ---
categories = {
    "Proximity-based": ["CBLOF", "HBOS", "KNN", "LOF", "DBSCAN"],
    "Probabilistic": ["ABOD", "GMM", "KDE", "ECOD", "COPOD"],
    "Ensembles": ["FB", "IForest", "LSCP", "INNE"],
    "Linear Models": ["MCD", "OCSVM", "PCA", "LMDD"]
}
key_map = {k: k.lower().replace(" ", "") for k in categories}

# --- Layout columns ---
col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])

# --- Dataset Selection ---
with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox("Choose Dataset", [
        "Bank", "BeijingClimate", "Breast-cancer-wisconsin", "CardioIsomap", "CoilDensmap",
        "Iris", "Nhanes", "ObesityDataSet", "PIRSensor"
    ], key="dataset_select")

    if dataset and st.session_state.get("dataset") != dataset:
        st.session_state["confirmed"] = False
        st.session_state["algorithms"] = {}
        st.session_state["heatmap_type"] = None
        st.session_state["colormap"] = None
        for group, algos in categories.items():
            for algo in algos:
                st.session_state[f"{group}-{algo}"] = False

        X = load_cached_dataset(dataset)
        X2d, xlim, ylim = get_pca_coords(X)
        st.session_state["pca"] = {"X2d": X2d, "xlim": xlim, "ylim": ylim}
        st.session_state["dataset"] = dataset

    if "pca" in st.session_state:
        X2d = st.session_state["pca"]["X2d"]
        xlim = st.session_state["pca"]["xlim"]
        ylim = st.session_state["pca"]["ylim"]
        fig, ax = plt.subplots()
        ax.scatter(X2d[:,0], X2d[:,1], s=6, color='#3b82f6', alpha=0.7, edgecolors='none')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"PCA Scatter of {dataset}")
        st.pyplot(fig, use_container_width=True)


algo_full_names = {
    "CBLOF":  "Clustering-Based Local Outlier Factor",
    "HBOS":   "Histogram-based Outlier Score",
    "KNN":    "K-Nearest Neighbors",
    "LOF":    "Local Outlier Factor",
    "DBSCA":  "Density-Based Spatial Clustering of Applications with Noise (DBSCAN)",
    "ABOD":   "Angle-Based Outlier Detection",
    "GMM":    "Gaussian Mixture Model",
    "KDE":    "Kernel Density Estimation",
    "ECOD":   "Empirical Cumulative Distribution for Outlier Detection",
    "COPOD":  "Copula-Based Outlier Detection",
    "FB":     "Feature Bagging",
    "IForest":"Isolation Forest",
    "LSCP":   "Locally Selective Combination of Parallel Outlier Ensembles",
    "INNE":   "Isolation-based Nearest-Neighbor Ensembles",
    "MCD":    "Minimum Covariance Determinant",
    "OCSVM":  "One-Class Support Vector Machine",
    "PCA":    "Principal Component Analysis",
    "LMDD":   "Deviation-based Outlier Detection (LMDD)"
}

# --- Algorithm Selection ---
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    inner = st.columns(len(categories))

    for i, (group, algos) in enumerate(categories.items()):
        with inner[i]:
            st.markdown(
                f"<div class='algo-group'><strong>{group}</strong></div>",
                unsafe_allow_html=True
            )
            selected_count = sum(
                st.session_state.get(f"{group}-{algo}", False)
                for algo in algos
            )

            for algo in algos:
                key = f"{group}-{algo}"
                full = algo_full_names.get(algo, "")
                checked = st.session_state.get(key, False)
                disabled = (selected_count >= 2 and not checked)

                st.checkbox(
                    label=algo,
                    key=key,
                    help=full,
                    disabled=disabled
                )

# --- Heatmap + Colormap ---
with col3:
    st.markdown("<div class='highlight-header'>Visualization Techniques</div>", unsafe_allow_html=True)

    heatmap_values = ["raw", "threshold", "interpolated", "binary", "ranked"]
    st.radio(
        label="",
        options=heatmap_values,
        key="heatmap_type",
        label_visibility="collapsed",
        format_func=lambda x: x.capitalize()
    )


colormap_options = ["viridis", "plasma", "terrain", "coolwarm", "turbo", "cividis"]

for cmap in colormap_options:
    key = f"cb_{cmap}"
    if key not in st.session_state:
        st.session_state[key] = False

def _select_colormap(cmap):
    for o in colormap_options:
        st.session_state[f"cb_{o}"] = False
    st.session_state[f"cb_{cmap}"] = True
    st.session_state["colormap"] = cmap


with col4:
    st.markdown(
        "<div class='highlight-header' style='text-align:center; margin-bottom:8px;'>Colormap</div>",
        unsafe_allow_html=True
    )

    colormap_options = ["viridis", "plasma", "terrain", "coolwarm", "turbo", "cividis"]


    for row in range(3):
        cb1_col, img1_col, cb2_col, img2_col = st.columns([0.15, 0.85, 0.15, 0.85])

        idx1 = row * 2
        if idx1 < len(colormap_options):
            cmap1 = colormap_options[idx1]
            key1 = f"cb_{cmap1}"

            with cb1_col:
                st.checkbox(
                    label="",
                    key=key1,
                    label_visibility="collapsed",
                    on_change=_select_colormap,
                    args=(cmap1,)
                )

            with img1_col:
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                fig, ax = plt.subplots(figsize=(1.0, 0.40), dpi=100)
                ax.imshow(gradient, aspect="auto", cmap=cmap1)
                ax.axis("off")

               
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0.2)
                ax.text(
                    0.5, -0.08,
                    cmap1,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=12
                )

                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        
        idx2 = row * 2 + 1
        if idx2 < len(colormap_options):
            cmap2 = colormap_options[idx2]
            key2 = f"cb_{cmap2}"

            with cb2_col:
                st.checkbox(
                    label="",
                    key=key2,
                    label_visibility="collapsed",
                    on_change=_select_colormap,
                    args=(cmap2,)
                )

            with img2_col:
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                fig, ax = plt.subplots(figsize=(1.0, 0.40), dpi=100)
                ax.imshow(gradient, aspect="auto", cmap=cmap2)
                ax.axis("off")

                fig.subplots_adjust(left=0, right=1, top=1, bottom=0.2)
                ax.text(
                    0.5, -0.08,
                    cmap2,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=12
                )

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# --- Explore Button ---
explore_disabled = not (
    st.session_state.get("dataset_select") and
    st.session_state.get("heatmap_type") and
    st.session_state.get("colormap") and
    all(
        sum([
            st.session_state.get(f"{group}-{algo}", False)
            for algo in algos
        ]) == 2 for group, algos in categories.items()
    )
)

_, mid, _ = st.columns([1, 1, 1])
with mid:
    st.button("Explore", key="continue_btn", disabled=explore_disabled,
        on_click=lambda: st.session_state.update({
            "confirmed": True,
            "algorithms": {
                key_map[group]: [
                    algo for algo in algos if st.session_state.get(f"{group}-{algo}", False)
                ] for group, algos in categories.items()
            }
        }))


# --- Visualization ---
if st.session_state.get("confirmed") and st.session_state.get("heatmap_type"):
    X = StandardScaler().fit_transform(load_cached_dataset(st.session_state["dataset"]))
    X2d = st.session_state["pca"]["X2d"]
    xlim = st.session_state["pca"]["xlim"]
    ylim = st.session_state["pca"]["ylim"]

    def render_heatmap(X2d, scores, method, cmap, title):
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        if method == "threshold":
            scores = (scores >= np.percentile(scores, 95)).astype(float)
        elif method == "binary":
            scores = (scores >= 0.5).astype(float)
        elif method == "ranked":
            scores = scores.argsort().argsort() / len(scores)

        gx, gy = np.linspace(xlim[0], xlim[1], 60), np.linspace(ylim[0], ylim[1], 60)
        xx, yy = np.meshgrid(gx, gy)
        interpolator = RBFInterpolator(X2d, scores, neighbors=20, smoothing=0.1)
        zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        if method in ["threshold", "interpolated", "ranked"]:
            zz = gaussian_filter(zz, sigma=1.5)

    
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, zz, levels=60, cmap=cmap)
        ax.scatter(X2d[:,0], X2d[:,1], s=4, c="black", alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(title, fontsize=12, color="black")
        fig.patch.set_facecolor("none")
        return fig

    cols = st.columns([1, 1, 1, 1, 0.3])
    cat_order = list(categories.keys())
    idx_map = {cat: i for i, cat in enumerate(cat_order)}

    for cat in cat_order:
        with cols[idx_map[cat]]:
            st.markdown(
                f"<div style='text-align:center; margin-bottom:4px;'><strong>{cat}</strong></div>",
                unsafe_allow_html=True
            )

            for algo in st.session_state["algorithms"].get(key_map[cat], []):
                try:
                    scores = get_scores_cached(algo, X)
                    fig = render_heatmap(
                        X2d,
                        scores,
                        st.session_state["heatmap_type"],
                        st.session_state["colormap"],
                        algo
                    )
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"{algo} failed: {e}")

    with cols[-1]:
        fig, ax = plt.subplots(figsize=(0.9, 2.5), dpi=100)
        gradient = np.linspace(0,1,256).reshape(256,1)
        ax.imshow(
            gradient[::-1], aspect="auto",
            cmap=st.session_state["colormap"],
            extent=[0,1,0,1]
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 1.02, "Outlier", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6)
        ax.text(0.5, -0.02, "Inlier", transform=ax.transAxes,
                ha="center", va="top", fontsize=6)
        for s in ax.spines.values(): s.set_visible(False)
        fig.subplots_adjust(left=0.4, right=0.6, top=1.1, bottom=0.05)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)

