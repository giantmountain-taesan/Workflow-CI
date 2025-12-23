import pandas as pd
import mlflow 
from sklearn.cluster import KMeans

# 1. Load data 
df = pd.read_csv('StudentPerformanceFactors_preprocessing.csv')
X = df.select_dtypes(include=['float64', 'int64'])

# 2. Training K-Means
n_clusters = 3

# PAKAI INI SAJA: Tanpa nama, tanpa ID, biar MLflow buat yang benar-benar baru
with mlflow.start_run():
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Manual logging untuk syarat Advanced
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("inertia", kmeans.inertia_)

    print("Berhasil!")
