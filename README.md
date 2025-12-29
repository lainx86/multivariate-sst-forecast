# Deep Blue Forecasting: Indonesian SST Anomaly Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Oceanography](https://img.shields.io/badge/Domain-Oceanography-teal)

## Project Overview
Proyek ini bertujuan untuk memprediksi **Anomali Suhu Permukaan Laut (SST)** di perairan Indonesia menggunakan pendekatan **Deep Learning (Long Short-Term Memory / LSTM)**. 

Berbeda dengan prediksi deret waktu biasa, model ini bersifat **Multivariate**. Model tidak hanya belajar dari data historis suhu lokal, tetapi juga memperhitungkan fenomena iklim global **El Niño-Southern Oscillation (ENSO)** melalui indeks **Niño 3.4** sebagai *predictor* eksternal.

Proyek ini menggabungkan pemahaman **Oseanografi Fisika** (Telekoneksi & Dinamika Iklim) dengan teknik **Computational Data Science**.

---

## Scientific Background
Mengapa memprediksi SST Indonesia itu penting?
* **Kesehatan Karang:** Anomali positif yang ekstrem menyebabkan *Coral Bleaching*.
* **Perikanan:** Suhu mempengaruhi migrasi ikan dan *upwelling*.
* **Telekoneksi Iklim:** Melalui **Sirkulasi Walker**, kondisi di Pasifik Tengah (El Niño/La Niña) mempengaruhi curah hujan dan suhu di Indonesia dengan jeda waktu (*time lag*) tertentu. Model ini memanfaatkan *lag* tersebut untuk meningkatkan akurasi prediksi.

---

## Tech Stack & Libraries
* **Core:** Python
* **Deep Learning:** PyTorch (LSTM Architecture)
* **Data Processing:** Xarray (NetCDF handling), Pandas, NumPy
* **Visualization:** Matplotlib

---

## Data Sources
Proyek ini menggunakan data iklim otentik dari NOAA (National Oceanic and Atmospheric Administration):

1.  **Target (Y): Indonesian SST**
    * **Sumber:** NOAA OISST V2 (High Resolution).
    * **Processing:** Data harian dipotong (*sliced*) untuk wilayah Indonesia (95°E - 141°E, 11°S - 6°N), di-resample ke bulanan, dan dihitung anomalinya (dikurangi klimatologi bulanan).
2.  **Feature (X): Niño 3.4 Index**
    * **Sumber:** NOAA ERSSTv5 (SST Anomaly di Pasifik Tengah).
    * **Fungsi:** Sebagai *leading indicator* untuk mendeteksi fase El Niño/La Niña.

---

## Methodology

### 1. Preprocessing Pipeline (preprocessing.py)
* **ETL (Extract, Transform, Load):** Mengambil data mentah `.nc` (NetCDF).
* **Spatial Aggregation:** Mengubah data 3D (Lat, Lon, Time) menjadi 1D Time Series (Time) dengan merata-ratakan seluruh wilayah Indonesia.
* **Deseasonalizing:** Menghapus siklus tahunan (musim) untuk mendapatkan sinyal anomali murni.

### 2. Model Architecture (modeling.py)
* **Type:** Multivariate LSTM (Long Short-Term Memory).
* **Input Shape:** `(Batch_Size, Sequence_Length, 2)` -> *2 Fitur: SST Indo & Niño 3.4*.
* **Lookback Window:** 12 Bulan (Model melihat 1 tahun ke belakang untuk memprediksi bulan depan).
* **Loss Function:** Mean Squared Error (MSE).

---

## Results & Analysis

### Actual vs Predicted Anomaly
*(Masukkan gambar grafik hasil model kamu di sini)*
![Model Result](image_4623ea.png)

**Insight Oseanografi:**
Grafik di atas menunjukkan kemampuan model dalam menangkap fenomena iklim:
* **Fase El Niño:** Menunjukkan periode di mana Indeks Niño 3.4 bernilai tinggi (El Niño Kuat).
* **Respon Model:** Garis prediksi menunjukkan penurunan suhu (anomali negatif) yang signifikan selama fase El Niño. Hal ini konsisten dengan teori bahwa El Niño menyebabkan *Indonesian Throughflow* melemah dan atmosfer di atas Indonesia menjadi lebih kering/dingin.

---

## How to Run
1.  **Clone Repository**
    ```bash
    git clone [https://github.com/username-kamu/nama-repo.git](https://github.com/username-kamu/nama-repo.git)
    cd nama-repo
    ```
2.  **Install Requirements**
    ```bash
    pip install xarray netCDF4 pandas numpy torch matplotlib
    ```
3.  **Run Preprocessing**
    ```bash
    python preprocessing.py
    ```
4.  **Train & Evaluate Model**
    ```bash
    python modeling.py
    ```
---

*Project ini dibuat sebagai bagian dari eksplorasi Data Science di bidang Oseanografi.*