import gradio as gr
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import tempfile
import os

pretrained_models = {
    "KMeans": joblib.load("kmeans_model.pkl"),
    "DBSCAN": joblib.load("dbscan_model.pkl"),
    "Agglomerative": joblib.load("agglo_model.pkl")
}

def clustering_app(file, model_choice, pkl_file):
    if file is None:
        return "⚠️ Please upload a dataset.", None, None

    try:
        df = pd.read_csv(file.name, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file.name, encoding="ISO-8859-1")

    df_num = df.select_dtypes(include=["float64", "int64"]).copy()
    if df_num.empty:
        return "⚠️ No numeric columns found in dataset.", None, None


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)


    if pkl_file is not None:
        model = joblib.load(pkl_file.name)
    else:
        model = pretrained_models[model_choice]

 
    try:
        labels = model.fit_predict(X_scaled)
    except AttributeError:
        labels = model.predict(X_scaled)

    df_num["Cluster"] = labels

    numeric_cols = df_num.columns[:-1] 
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
        return "⚠️ Not enough features for visualization (need at least 2).", None, None

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_pdf.name)


    data = [df_num.columns.tolist()] + df_num.values.tolist()  

    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ])
    table.setStyle(style)

    doc.build([table])

    return "✅ Clustering completed!", fig, tmp_pdf.name



with gr.Blocks() as demo:
    gr.Markdown("## 🌀 Clustering App with Model Selection & PKL Upload")

    with gr.Row():
        file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
        pkl_input = gr.File(label="Upload Custom Model (.pkl)", file_types=[".pkl"])
    
    model_choice = gr.Dropdown(choices=list(pretrained_models.keys()), value="KMeans", label="Choose Model")

    run_button = gr.Button("🚀 Run Clustering")

    output_text = gr.Textbox(label="Status")
    output_plot = gr.Plot(label="Cluster Visualization")
    output_file = gr.File(label="📥 Download Clustered Data (PDF)")

    run_button.click(
        clustering_app,
        inputs=[file_input, model_choice, pkl_input],
        outputs=[output_text, output_plot, output_file]
    )

if __name__ == "__main__":
    demo.launch()
