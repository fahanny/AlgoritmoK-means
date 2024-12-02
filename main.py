import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suprimir o aviso de núcleos físicos do joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# Configurações para gráficos
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Caminhos relativos
base_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório do script
dataset_folder = os.path.join(base_dir, "UCI HAR Dataset")
features_path = os.path.join(dataset_folder, "features.txt")
X_train_path = os.path.join(dataset_folder, "train", "X_train.txt")
y_train_path = os.path.join(dataset_folder, "train", "y_train.txt")

# Verificação de arquivos
for path in [features_path, X_train_path, y_train_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

# Carregar dados
features = pd.read_csv(features_path, sep=r"\s+", header=None, usecols=[1], names=["feature"])
X_train = pd.read_csv(X_train_path, sep=r"\s+", header=None)
y_train = pd.read_csv(y_train_path, sep=r"\s+", header=None, names=["activity"])

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Reduzir a dimensionalidade com PCA para 50 componentes (ao invés de 561)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)

# Método do Cotovelo (Elbow Method) com MiniBatchKMeans para performance
inertia = []
K = range(2, 21)  # Testar para K de 2 a 20
for k in K:
    kmeans = MiniBatchKMeans(n_clusters=k, init="k-means++", random_state=42, n_init=5)  # MiniBatchKMeans para performance
    kmeans.fit(X_train_pca)
    inertia.append(kmeans.inertia_)

# Gráfico do Método do Cotovelo
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, marker='o', label="Inércia")
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inércia")
plt.grid()
plt.legend()
plt.show()

# Escolher K baseado no cotovelo (olhar o gráfico e escolher o K no ponto de inflexão)
optimal_k_cotovelo = K[np.argmax(inertia)]  # Pegando o K com maior inércia (cotovelo)
print(f"O número ideal de clusters (K) pelo cotovelo é: {optimal_k_cotovelo}")

# Aplicação do MiniBatchKMeans com K do método do cotovelo
kmeans_cotovelo = MiniBatchKMeans(n_clusters=optimal_k_cotovelo, init="k-means++", random_state=42, n_init=5)
clusters_cotovelo = kmeans_cotovelo.fit_predict(X_train_pca)

# PCA para redução de dimensionalidade (2D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train_pca)

# Visualização em 2D com PCA para clusters do cotovelo
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters_cotovelo, cmap='viridis', s=50)
plt.title(f"Clusters em 2D com PCA (K={optimal_k_cotovelo} - Cotovelo)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(scatter, label="Cluster")
plt.grid()
plt.savefig("clusters_cotovelo.png")  # Salvar o gráfico
plt.show()

# Método do Silhouette (Cálculo do Silhouette Score)
silhouettes = []
for k in K:
    kmeans = MiniBatchKMeans(n_clusters=k, init="k-means++", random_state=42, n_init=5)
    kmeans.fit(X_train_pca)
    silhouettes.append(silhouette_score(X_train_pca, kmeans.labels_))

# Gráfico do Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(K, silhouettes, marker='o', label="Silhouette Score", color='orange')
plt.title("Silhouette Score para diferentes K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.legend()
plt.savefig("silhouette_score.png")  # Salvar o gráfico
plt.show()

# Escolher K automaticamente com base no maior Silhouette Score
optimal_k_silhouette = K[np.argmax(silhouettes)]  # Pegando o K com maior Silhouette Score
print(f"O número ideal de clusters (K) pelo Silhouette Score é: {optimal_k_silhouette}")

# Aplicação do MiniBatchKMeans com K do Silhouette Score
kmeans_silhouette = MiniBatchKMeans(n_clusters=optimal_k_silhouette, init="k-means++", random_state=42, n_init=5)
clusters_silhouette = kmeans_silhouette.fit_predict(X_train_pca)

# Visualização em 2D com PCA para clusters do Silhouette Score
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters_silhouette, cmap='plasma', s=50)
plt.title(f"Clusters em 2D com PCA (K={optimal_k_silhouette} - Silhouette)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(scatter, label="Cluster")
plt.grid()
plt.savefig("clusters_silhouette.png")  # Salvar o gráfico
plt.show()

# Análise de Estabilidade e Repetições - Usando K do Cotovelo
results = []
for _ in range(5):  # Rodar múltiplas vezes para verificar a estabilidade (reduzimos de 10 para 5)
    kmeans = MiniBatchKMeans(n_clusters=optimal_k_cotovelo, init="k-means++", random_state=42, n_init=5)
    kmeans.fit(X_train_pca)
    results.append(kmeans.labels_)

# Verificar consistência entre as execuções
consistency = np.mean([np.array_equal(results[0], result) for result in results])
print(f"Consistência entre execuções (Cotovelo): {consistency:.2f}")

# Análise de Estabilidade e Repetições - Usando K do Silhouette
results_silhouette = []
for _ in range(5):  # Rodar múltiplas vezes para verificar a estabilidade (reduzimos de 10 para 5)
    kmeans = MiniBatchKMeans(n_clusters=optimal_k_silhouette, init="k-means++", random_state=42, n_init=5)
    kmeans.fit(X_train_pca)
    results_silhouette.append(kmeans.labels_)

# Verificar consistência entre as execuções
consistency_silhouette = np.mean([np.array_equal(results_silhouette[0], result) for result in results_silhouette])
print(f"Consistência entre execuções (Silhouette): {consistency_silhouette:.2f}")

# Análise das características de cada cluster
df_clusters = pd.DataFrame(X_train_scaled, columns=features["feature"])  # Dados originais
df_clusters["Cluster_Cotovelo"] = clusters_cotovelo  # Clusters do cotovelo
df_clusters["Cluster_Silhouette"] = clusters_silhouette  # Clusters do Silhouette
df_clusters["Activity"] = y_train.values  # Atividades originais

# Estatísticas descritivas por cluster
print("\nEstatísticas por Cluster (Cotovelo):")
print(df_clusters.groupby("Cluster_Cotovelo").mean())  # Média das variáveis por cluster

print("\nEstatísticas por Cluster (Silhouette):")
print(df_clusters.groupby("Cluster_Silhouette").mean())  # Média das variáveis por cluster

# Verificar a distribuição de atividades por cluster
print("\nDistribuição de Atividades por Cluster (Cotovelo):")
print(df_clusters.groupby(["Cluster_Cotovelo", "Activity"]).size())

print("\nDistribuição de Atividades por Cluster (Silhouette):")
print(df_clusters.groupby(["Cluster_Silhouette", "Activity"]).size())

