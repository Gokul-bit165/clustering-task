# 🌀 Clustering App with Gradio  

An interactive **Clustering Web Application** built using **Python, Scikit-learn, Plotly, and Gradio**.  
This app allows users to:  
- Upload their own dataset (CSV)  
- Choose a clustering algorithm (KMeans, DBSCAN, Agglomerative)  
- Optionally upload a pretrained `.pkl` model  
- Visualize clusters in **2D/3D interactive plots**  

---

## 🚀 Features  
✅ Upload any CSV dataset  
✅ Upload or use pre-trained `.pkl` clustering models  
✅ Choose clustering algorithm from dropdown  
✅ Real-time **interactive 3D/2D plots** with Plotly  
✅ Automatic preprocessing (numeric columns + scaling)  
✅ Clustered results downloadable as CSV  

---

## ⚡ Installation  

```bash
# Clone repo
git clone https://github.com/Gokul-bit165/clustering-task.git
cd clustering-app

# Create environment (optional)
conda create -n clustering python=3.10 -y
conda activate clustering

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 File Structure  

```
clustering-app/
│── app.py                 # Main Gradio app  
│── kmeans_model.pkl       # Pre-trained KMeans model  
│── dbscan_model.pkl       # Pre-trained DBSCAN model  
│── agglo_model.pkl        # Pre-trained Agglomerative model  
│── requirements.txt       # Dependencies  
│── README.md              # Project documentation  
```

---

## ▶️ Run App  

```bash
python app.py
```

Local URL: `http://127.0.0.1:7860`  
Public Share (if enabled): `https://huggingface.co/spaces/GokulV/clustering/`  

---

## 🎮 Usage  

1. Upload your **CSV dataset**  
2. (Optional) Upload a **pretrained `.pkl` model**  
3. Select algorithm: **KMeans / DBSCAN / Agglomerative**  
4. Click **Run Clustering**  
5. See interactive **3D cluster plot**  
6. Download results with cluster labels  

---

## 📊 Example Output  

### 🔹 KMeans Example  
![KMeans Example](rfm_outputs/kmeans.png)  

### 🔹 DBSCAN Example  
![DBSCAN Example](rfm_outputs/dbscan.png)  

### 🔹 Agglomerative Example  
![Agglomerative Example](rfm_outputs/heirarichal.png)  

---

## 🌍 Live Demo  
🔗 [Try it here](https://huggingface.co/spaces/GokulV/clustering/)  

---

## ✅ Conclusion  
This app makes clustering **simple, visual, and interactive**.  
It is ideal for:  
- Customer Segmentation  
- Market Basket Analysis  
- Exploratory Data Analysis (EDA)  
- Unsupervised Learning Projects  

## Evaluation metrics:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Silhouette Score ↑</th>
      <th>Calinski-Harabasz ↑</th>
      <th>Davies-Bouldin ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KMeans</td>
      <td>0.439315</td>
      <td>2417.851924</td>
      <td>0.971942</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hierarchical</td>
      <td>0.502359</td>
      <td>586.403492</td>
      <td>0.614819</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DBSCAN</td>
      <td>0.226870</td>
      <td>948.337929</td>
      <td>1.614359</td>
    </tr>
  </tbody>
</table>
</div>