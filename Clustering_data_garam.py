from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
from scipy.io.arff import loadarff
from io import StringIO, BytesIO
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Dataset", "Pre-processing",
                 "Modeling", "Evaluasi"],
        icons=["house", "clipboard-data", "wrench",
               "diagram-3", "pc-display-horizontal"],
        menu_icon="cast",
        default_index=0,
    )

df = pd.read_csv(
    "https://raw.githubusercontent.com/UswatunChasanah/dataset/main/data%20garam.csv")

df['Grade'] = pd.Categorical(df["Grade"])
df["Grade"] = df["Grade"].cat.codes

hapus =['Grade','Data']
data = df.drop(hapus,axis=1)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Inisialisasi objek MinMaxScaler
scaler = MinMaxScaler()

# Fit transform data menggunakan MinMaxScaler
scaled_data = scaler.fit_transform(data)

# Konversi data yang telah diubah ke dataframe
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

from sklearn.cluster import KMeans
import numpy as np

def kmeans_clustering(data, n_clusters, max_iterations=100):
  # Membuat objek KMeans dengan jumlah klaster yang ditentukan
  kmeans = KMeans(n_clusters=n_clusters)
        
  # Melatih model k-means
  kmeans.fit(data)
        
  # Mendapatkan label klaster untuk setiap sampel
  labels = kmeans.labels_
        
  # Mendapatkan pusat klaster
  centroids = kmeans.cluster_centers_
        
  # Inisialisasi flag perubahan klaster
  klaster_changed = True
        
  # Melakukan perulangan hingga tidak ada lagi perubahan klaster atau mencapai batas iterasi maksimum
  iteration = 0
  while klaster_changed and iteration < max_iterations:
    # Menghitung jarak antara setiap sampel dengan pusat klaster
    distances = kmeans.transform(data)
            
    # Mendapatkan indeks klaster terdekat untuk setiap sampel
    closest_clusters = np.argmin(distances, axis=1)
            
    # Memeriksa apakah ada perubahan klaster
    if not np.array_equal(labels, closest_clusters):
      # Terdapat perubahan klaster
      klaster_changed = True
                
      # Mengupdate label klaster
      labels = closest_clusters
                
      # Mengupdate pusat klaster
      centroids = kmeans.cluster_centers_
                
      # Melatih model k-means dengan inisialisasi centroid baru
      kmeans = KMeans(n_clusters=n_clusters, init=centroids)
      kmeans.fit(data)
    else:
      # Tidak ada perubahan klaster
      klaster_changed = False
            
    # Meningkatkan iterasi
    iteration += 1
        
  # Mengembalikan label klaster dan pusat klaster
  return labels, centroids

labels, centroids = kmeans_clustering(scaled_df, n_clusters=4)

if selected == "Home":
    st.write("""
    # Clustering Data Garam 
    ## (Studi Kasus : PT. Garam Sumenep)
    """
             )
    image = Image.open('garam.jpg')
    st.image(image)
if selected == "Dataset":
    st.markdown("<h1 style='text-align: center;'>Data Set</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Data Garam PT. Garam Sumenep</h3>",
                unsafe_allow_html=True)
    st.dataframe(df)

if selected == "Pre-processing":
    st.markdown("<h1 style='text-align: center;'>Pre-Processing</h1>",
                unsafe_allow_html=True)

    st.write("""
    # Normalisasi Data
    Normalisasi digunakan untuk mengubah nilai kolom numerik dalam himpunan data untuk menggunakan skala umum, tanpa mendistorsi perbedaan dalam rentang nilai atau kehilangan informasi.
    """)

    # Menampilkan dataframe hasil scaling
    st.dataframe(scaled_df)


if selected == "Modeling":
  st.markdown("<h1 style='text-align: center;'>Hasil Modeling</h1>",
                unsafe_allow_html=True)
  st.write("Modelling dilakukan menggunakan 3 metode yaitu K-Means, AHC dan DBSCAN")

  st.markdown("<h2>Metode K-Means</h2>",
                unsafe_allow_html=True)

  unique_labels, counts = np.unique(labels, return_counts=True)

  st.markdown("<h4>Jumlah data disetiap Klaster</h4>",
                unsafe_allow_html=True)

  # Menampilkan jumlah data setiap label
  for label, count in zip(unique_labels, counts):
      st.write(f"Jumlah data dengan label {label}: {count}")

  st.markdown("<h4>Data Garam Yang Sudah Di Clustering</h4>",
                unsafe_allow_html=True)

  df['Klaster']=pd.DataFrame({'Klaster':labels})
  df[['Kadar Air ','Tak Larut','Kalsium','Magnesium','Sulfat ','NaCl (wb)','NaCl (db)','Klaster']]
    
  import matplotlib.pyplot as plt
  import numpy as np

  def plot_clustering(data, labels, centroids):
    # Mendapatkan jumlah klaster
    n_clusters = len(centroids)
    
    # Membuat scatter plot untuk setiap klaster
    for cluster in range(n_clusters):
      # Memfilter data berdasarkan label klaster
      cluster_data = data[labels == cluster]
      
      # Memplot data klaster
      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
    
    # Memplot pusat klaster
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
    
    # Menambahkan label dan judul plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot Hasil Clustering')
    
    # Menampilkan legenda
    plt.legend()
    
    # Mengambil plot sebagai objek BytesIO
    plot_bytes = BytesIO()
    plt.savefig(plot_bytes, format='png')
    plot_bytes.seek(0)
    
    # Menampilkan plot menggunakan Streamlit
    st.image(plot_bytes)


  # Memanggil fungsi plot_clustering
  plot_clustering(scaled_df.values, labels, centroids)

  st.markdown("<h2>Metode AHC</h2>", unsafe_allow_html=True)
  from sklearn.cluster import AgglomerativeClustering
  from sklearn.metrics.pairwise import pairwise_distances

  # Pilih metrik jarak
  distance_metric = 'euclidean'

  # Inisialisasi cluster
  n_clusters = 4
  clustering = AgglomerativeClustering(n_clusters=n_clusters)

  # Hitung matriks jarak
  distance_matrix = pairwise_distances(scaled_df, metric=distance_metric)

  # 5. Gabungkan kluster terdekat
  clustering.fit(distance_matrix)

  # 6. Ulangi langkah 5 jika diperlukan

  # 7. Interpretasi hasil
  labels = clustering.labels_
  for i in range(n_clusters):
      cluster_i_indices = [index for index, label in enumerate(labels) if label == i]
      # st.write(f'Jumlah data dengan label {i}: {cluster_i_indices}')
  unique_labels, counts = np.unique(labels, return_counts=True)

  # Menampilkan jumlah data setiap label
  st.markdown("<h4>Jumlah data disetiap Klaster</h4>",
                unsafe_allow_html=True)
  for label, count in zip(unique_labels, counts):
      st.write(f"Jumlah data dengan label {label}: {count}")

  st.markdown("<h4>Data Garam Yang Sudah Di Clustering Menggunakan AHC</h4>",
                unsafe_allow_html=True)
  df['Klaster']=pd.DataFrame({'Klaster':labels})
  df[['Kadar Air ','Tak Larut','Kalsium','Magnesium','Sulfat ','NaCl (wb)','NaCl (db)','Klaster']]

  import streamlit as st
  import matplotlib.pyplot as plt
  from scipy.cluster.hierarchy import dendrogram, linkage
  from sklearn.metrics.pairwise import pairwise_distances

  def plot_dendrogram(scaled_df, distance_metric, n_clusters):
      # Hitung matriks jarak
      distance_matrix = pairwise_distances(scaled_df, metric=distance_metric)

      # Lakukan klasterisasi dengan AHC menggunakan linkage matrix
      linkage_matrix = linkage(distance_matrix, method='ward')

      # Plot dendrogram
      plt.figure(figsize=(10, 5))
      dendrogram(linkage_matrix)
      plt.title('Dendrogram Visualization')
      plt.xlabel('Data Index')
      plt.ylabel('Distance')
      plt.show()
      st.pyplot(plt.gcf())

  # Panggil fungsi plot_dendrogram di Streamlit
  # plt.title('Dendrogram Visualization')
  # st.write('Masukkan argumen untuk visualisasi dendrogram:')
  # distance_metric = st.selectbox('Metrik Jarak', ['euclidean', 'manhattan', 'cosine'])
  # n_clusters = st.slider('Jumlah Cluster', min_value=2, max_value=10, value=4)

  # Panggil fungsi plot_dendrogram dengan argumen yang diberikan
  plot_dendrogram(scaled_df, distance_metric, n_clusters)





  st.markdown("<h2>Metode DBSCAN</h2>", unsafe_allow_html=True)
  from sklearn.cluster import DBSCAN

  # Inisialisasi dan fit model DBSCAN
  dbscan = DBSCAN(eps=0.5, min_samples=3)
  dbscan.fit(scaled_df)

  # Menampilkan hasil klasterisasi
  labels = dbscan.labels_
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  # print("Jumlah klaster:", n_clusters)
  # print("Label klaster:", labels)

  unique_labels, counts = np.unique(labels, return_counts=True)

  # Menampilkan jumlah data setiap label
  st.markdown("<h4>Jumlah data disetiap Klaster</h4>",
                unsafe_allow_html=True)
  # Menampilkan jumlah data setiap label
  for label, count in zip(unique_labels, counts):
      st.write(f"Jumlah data dengan label {label}: {count}")

  st.markdown("<h4>Data Garam Yang Sudah Di Clustering Menggunakan DBSCAN</h4>",
                unsafe_allow_html=True)
  df['Klaster']=pd.DataFrame({'Klaster':labels})
  df[['Kadar Air ','Tak Larut','Kalsium','Magnesium','Sulfat ','NaCl (wb)','NaCl (db)','Klaster']]




if selected == "Evaluasi":
  st.markdown("<h1 style='text-align: center;'>Evaluasi</h1>",
                unsafe_allow_html=True)

  st.markdown("<h4>Hasil Kualitas Clustering Menggunakan Metode K-Means</h4>",
                unsafe_allow_html=True)
  
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  import numpy as np

  # 2. Pilih jumlah klaster yang ingin diuji
  k_values = [4]

  # 3. Hitung Silhouette Coefficient untuk setiap jumlah klaster
  for k in k_values:
      kmeans = KMeans(n_clusters=k)
      kmeans.fit(data)
      labels = kmeans.labels_
      silhouette_avg = silhouette_score(data, labels)
      st.write(f'Silhouette Coefficient : {silhouette_avg}')

  st.markdown("<h4>Hasil Kualitas Clustering Menggunakan Metode AHC</h4>",
                unsafe_allow_html=True)
  from sklearn.cluster import AgglomerativeClustering
  from sklearn.metrics import silhouette_score
  import numpy as np


  # 2. Pilih jumlah klaster yang ingin diuji
  k_values = [4]

  # 3. Hitung Silhouette Coefficient untuk setiap jumlah klaster
  for k in k_values:
      agglomerative = AgglomerativeClustering(n_clusters=k)
      labels = agglomerative.fit_predict(data)
      silhouette_avg = silhouette_score(data, labels)
      st.write(f'Silhouette Coefficient : {silhouette_avg}')
  
  st.markdown("<h4>Hasil Kualitas Clustering Menggunakan Metode AHC</h4>",
                unsafe_allow_html=True)
  from sklearn.cluster import DBSCAN
  from sklearn.metrics import silhouette_score
  import numpy as np

  # 2. Tentukan parameter DBSCAN
  epsilon = 0.5  # Radius jangkauan (epsilon)
  min_samples = 3  # Jumlah minimum sampel dalam radius epsilon

  # 3. Lakukan klasterisasi dengan DBSCAN
  dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
  labels = dbscan.fit_predict(data)

  # 4. Hitung Silhouette Coefficient
  silhouette_avg = silhouette_score(data, labels)

  # 5. Cetak Silhouette Coefficient
  st.write(f"Silhouette Coefficient: {silhouette_avg}")
