import gradio as gr
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load your pre-trained models once
pretrained_models = {
    "KMeans": joblib.load("kmeans_model.pkl"),
    "DBSCAN": joblib.load("dbscan_model.pkl"),
    "Agglomerative": joblib.load("agglo_model.pkl")
}

def clustering_app(file, model_choice, pkl_file):
    if file is None:
        return "⚠️ Please upload a dataset.", None

    # Load dataset safely with fallback encoding
    try:
        df = pd.read_csv(file.name, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file.name, encoding="ISO-8859-1")

    # Select numeric features
    df_num = df.select_dtypes(include=["float64", "int64"]).copy()
    if df_num.empty:
        return "⚠️ No numeric columns found in dataset.", None

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    # Load model (priority: user uploaded → fallback to pretrained)
    if pkl_file is not None:
        model = joblib.load(pkl_file.name)
    else:
        model = pretrained_models[model_choice]

    # Predict / fit_predict depending on model type
    try:
        labels = model.fit_predict(X_scaled)
    except AttributeError:
        labels = model.predict(X_scaled)

    # Add cluster labels
    df_num["Cluster"] = labels

    # Choose features for visualization
    numeric_cols = df_num.columns[:-1]  # exclude Cluster col
    if len(numeric_cols) >= 3:
        cols = numeric_cols[:3]
        fig = px.scatter_3d(
            df_num,
            x=cols[0], y=cols[1], z=cols[2],
            color="Cluster",
            title=f"3D Clustering with {model_choice}"
        )
    elif len(numeric_cols) == 2:
        cols = numeric_cols[:2]
        fig = px.scatter(
            df_num,
            x=cols[0], y=cols[1],
            color="Cluster",
            title=f"2D Clustering with {model_choice}"
        )
    else:
        return "⚠️ Not enough features for visualization (need at least 2).", None

    return "✅ Clustering completed!", fig



# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌀 Clustering App with Model Selection & PKL Upload")

    with gr.Row():
        file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
        pkl_input = gr.File(label="Upload Custom Model (.pkl)", file_types=[".pkl"])
    
    model_choice = gr.Dropdown(choices=list(pretrained_models.keys()), value="KMeans", label="Choose Model")

    run_button = gr.Button("🚀 Run Clustering")

    output_text = gr.Textbox(label="Status")
    output_plot = gr.Plot(label="Cluster Visualization")

    run_button.click(
        clustering_app,
        inputs=[file_input, model_choice, pkl_input],
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch()
