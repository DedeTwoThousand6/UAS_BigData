# Market Basket Analysis for Cross-Genre Content Recommendation using Apache Spark MLlib

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi **Market Basket Analysis (MBA)** menggunakan algoritma **FP-Growth** pada **Apache Spark MLlib** untuk membangun sistem **Cross-Genre Content Recommendation** berdasarkan riwayat tontonan pengguna.

Dataset yang digunakan berasal dari **Video.com** dengan ukuran sekitar **7,1 juta record** yang tersimpan pada **HDFS**. Analisis dilakukan secara terdistribusi menggunakan Apache Spark sehingga mampu memproses data dalam skala besar secara efisien.

---

## 🎯 Tujuan

Menganalisis pola kebiasaan menonton pengguna untuk menemukan hubungan antar kategori konten sehingga dapat memberikan rekomendasi seperti:

> **"User yang menonton genre X biasanya juga menonton genre Y."**

Contoh:

- TV Show → Movies
- Kids → TV Show
- Champions → Sports

---

## 🛠️ Teknologi yang Digunakan

- Python 3
- Apache Spark 3.1.1
- Spark MLlib
- FP-Growth
- Spark SQL
- Hadoop HDFS
- Docker

---

## 📂 Dataset

Dataset Video.com

Jumlah data:

- **7.127.384 baris**
- **41 kolom**

Kolom utama yang digunakan:

| Kolom | Keterangan |
|--------|------------|
| hash_watcher_id | ID unik pengguna |
| category_name | Kategori konten yang ditonton |

---

## ⚙️ Tahapan Pengerjaan

### 1. Load Dataset

Dataset dibaca langsung dari HDFS menggunakan Spark.

```python
spark.read.csv(...)
```

---

### 2. Exploratory Data Analysis (EDA)

Melakukan analisis awal terhadap dataset.

- Jumlah record
- Jumlah kolom
- Jumlah user unik
- Struktur dataset

---

### 3. Data Preprocessing

Tahapan preprocessing meliputi:

- Menghapus data NULL
- Menghapus kategori tidak valid
- Menghapus duplicate user-category
- Menghapus kategori long-tail (< 50 user)
- Menghapus user outlier menggunakan P99
- Normalisasi nama kategori

---

### 4. Transformasi Market Basket

Data diubah menjadi transaksi per pengguna menggunakan:

```python
groupBy(hash_watcher_id)
collect_set(category_name)
```

Contoh:

| User | Basket |
|------|--------|
| U1 | [movies, tv show, entertainment] |
| U2 | [sports, champions] |
| U3 | [kids, vlog] |

---

### 5. Train-Test Split

Dataset basket dibagi menjadi:

- Training : 80%
- Testing : 20%

---

### 6. FP-Growth

Model dibangun menggunakan Spark MLlib.

Parameter:

```text
minSupport = 0.001
minConfidence = 0.10
```

---

### 7. Filtering Association Rules

Rule difilter menggunakan:

- Lift ≥ 1.2
- Support Count ≥ 30

Agar rule yang dihasilkan benar-benar bermakna.

---

### 8. Cross-Genre Recommendation

Menampilkan rekomendasi berdasarkan Association Rules terbaik.

Contoh hasil:

| Jika Menonton | Direkomendasikan |
|---------------|------------------|
| Champions | Sports |
| TV Show | Movies |
| Movies | TV Show |
| Kids | TV Show |
| Kids | Vlog |

---

### 9. Evaluasi Model

Model dievaluasi menggunakan:

- Support
- Confidence
- Lift
- Coverage
- Hit Rate

---

## 📊 Hasil Pengujian

Dataset setelah preprocessing:

| Keterangan | Nilai |
|------------|-------|
| Data awal | 7.127.384 |
| Data setelah preprocessing | 3.719.246 |
| Basket user | 258.885 |
| Training | 207.219 |
| Testing | 51.666 |

---

## 📈 Hasil FP-Growth

| Parameter | Nilai |
|------------|-------|
| Frequent Itemsets | 99 |
| Association Rules | 107 |
| Rule Berkualitas | 22 |

---

## 🎯 Contoh Rekomendasi

| Antecedent | Consequent | Confidence | Lift |
|------------|------------|------------|------|
| Champions | Sports | 53.52% | 4.311 |
| TV Show | Movies | 63.06% | 2.904 |
| Movies | TV Show | 36.80% | 2.904 |
| Kids | TV Show | 28.73% | 2.267 |
| Kids | Vlog | 15.27% | 2.235 |

---

## 📊 Evaluasi

| Metrik | Hasil |
|---------|-------|
| Coverage | 99.90% |
| Hit Rate | 85.20% |

Interpretasi:

- Coverage tinggi menunjukkan model mampu memberikan rekomendasi kepada hampir seluruh pengguna.
- Hit Rate tinggi menunjukkan rekomendasi yang diberikan relevan dengan perilaku pengguna pada data uji.

---

## 📁 Struktur Project

```
market_basket_analysis/
│
├── market_basket_fp_growth.py
├── README.md
└── dataset/
```

---

## ▶️ Menjalankan Program

Masuk ke container Spark:

```bash
docker exec -it spark-master bash
```

Jalankan program:

```bash
/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/root/market_basket_fp_growth.py
```

---

## 📌 Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi **Cross-Genre Content Recommendation** menggunakan algoritma **FP-Growth** pada Apache Spark MLlib.

Sebanyak **22 association rules berkualitas** berhasil ditemukan dari jutaan data riwayat tontonan pengguna. Rule dengan nilai Lift tertinggi menunjukkan hubungan yang kuat antar kategori konten sehingga dapat dimanfaatkan sebagai dasar sistem rekomendasi pada platform video streaming.

Selain menghasilkan nilai **Support**, **Confidence**, dan **Lift**, model juga memperoleh **Coverage sebesar 99,90%** dan **Hit Rate sebesar 85,20%**, yang menunjukkan bahwa rekomendasi yang dihasilkan mampu menjangkau hampir seluruh pengguna dan memiliki tingkat relevansi yang tinggi.

---

## 👨‍💻 Anggota Kelompok

**Dede Jamaludin**
**Hasbi Baihaqi**
**Danar Iswara**
**Syifa Kanita Putri**
**Ari Mauludin**

Tugas UAS Big Data

Topik 3 — Market Basket Analysis for Cross-Genre Content Recommendation

Apache Spark MLlib