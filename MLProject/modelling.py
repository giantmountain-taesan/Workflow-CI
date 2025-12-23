import pandas as pd
import mlflow 
from sklearn.cluster import KMeans

# 1. Load data 
file_path = 'Membangun_model/StudentPerformanceFactors_preprocessing.csv'
df = pd.read_csv(file_path)

# Pastikan hanya mengambil kolom angka untuk KMeans
X = df.select_dtypes(include=['float64', 'int64'])

# 2. Aktifkan Autolog
mlflow.autolog()

# 3. Training K-Means
n_clusters = 3
with mlflow.start_run(run_name="KMeans_Baseline"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    print(" Berhasil! Sekarang ketik 'mlflow ui' di terminal.")