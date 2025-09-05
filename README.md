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
Public Share (if enabled): `https://xxxxx.gradio.live`  

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
![KMeans Example](images/kmeans_plot.png)  

### 🔹 DBSCAN Example  
![DBSCAN Example](images/dbscan_plot.png)  

### 🔹 Agglomerative Example  
![Agglomerative Example](images/agglo_plot.png)  

---

## 🌍 Live Demo  
🔗 [Try it here](https://your-deployed-link.com)  

---

## ✅ Conclusion  
This app makes clustering **simple, visual, and interactive**.  
It is ideal for:  
- Customer Segmentation  
- Market Basket Analysis  
- Exploratory Data Analysis (EDA)  
- Unsupervised Learning Projects  
