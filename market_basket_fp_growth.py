#!/usr/bin/env python3
# ==============================================================================
# UAS BIG DATA - APACHE SPARK MLLIB (VERSI DITINGKATKAN)
# TOPIK 3: Market Basket Analysis for Cross-Genre Content Recommendation
# ==============================================================================

import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, StringType
from pyspark.ml.fpm import FPGrowth


# ------------------------------------------------------------------------------
# FUNGSI BANTU
# ------------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


start_time = time.time()


def elapsed_since_start() -> float:
    return time.time() - start_time


# ------------------------------------------------------------------------------
# 1. SPARK SESSION
# ------------------------------------------------------------------------------
print_header("INISIALISASI SPARK SESSION")

spark = (
    SparkSession.builder
    .appName("MarketBasketAnalysis_CrossGenreRecommendation_v2")
    .config("spark.sql.shuffle.partitions", "200")   # diturunkan lagi setelah data mengecil (lihat bagian FPGrowth)
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("SparkSession berhasil dibuat.")
print(f"Spark version : {spark.version}")
print(f"App Name      : {spark.sparkContext.appName}")
print(f"Master        : {spark.sparkContext.master}")


# ------------------------------------------------------------------------------
# 2. LOAD DATASET DARI HDFS
# ------------------------------------------------------------------------------
print_header("LOAD DATASET DARI HDFS")

DATA_PATH = "hdfs://namenode:9000/shared/data/videodotcom_big.csv"
print(f"Membaca dataset dari: {DATA_PATH}")

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "false")
    .option("escape", "\"")
    .csv(DATA_PATH)
)

print("Dataset berhasil dimuat dari HDFS.")


# ------------------------------------------------------------------------------
# 3. INFORMASI DATASET
# ------------------------------------------------------------------------------
print_header("INFORMASI DATASET")

n_rows = df_raw.count()  # trigger cache pertama kali di sini
n_cols = len(df_raw.columns)

print(f"Jumlah baris  : {n_rows:,}")
print(f"Jumlah kolom  : {n_cols}")
print("\nSchema dataset:")
df_raw.printSchema()


# ------------------------------------------------------------------------------
# 4. EDA
# ------------------------------------------------------------------------------
print_header("EDA")

os_col_candidates = [c for c in df_raw.columns if c.lower() in ("os", "operating_system")]

print("\n[EDA] Jumlah user unik (hash_watcher_id), estimasi cepat (approx_count_distinct):")
if "hash_watcher_id" in df_raw.columns:
    n_unique_users_approx = df_raw.select(
        F.approx_count_distinct("hash_watcher_id").alias("n_unique_users_approx")
    ).first()["n_unique_users_approx"]
    print(f"Jumlah user unik (approx): {n_unique_users_approx:,}")
else:
    print("Kolom 'hash_watcher_id' tidak ditemukan dalam dataset.")



# ------------------------------------------------------------------------------
# 5. PREPROCESSING (DIPERKETAT)
# ------------------------------------------------------------------------------
print_header("PREPROCESSING")

INVALID_CATEGORY_VALUES = {
    "true", "false", "null", "none", "nan", "undefined", "unknown", "n/a", "-", ""
}

# --- Ringkasan penyebab baris invalid dalam SATU pass (bukan 4 count terpisah) --
df_flagged = df_raw.withColumn("category_name_norm", F.lower(F.trim(F.col("category_name")))) \
    .withColumn("hash_watcher_id_norm", F.trim(F.col("hash_watcher_id")))

drop_summary = df_flagged.select(
    F.sum(F.when(F.col("hash_watcher_id").isNull(), 1).otherwise(0)).alias("null_watcher_id"),
    F.sum(F.when(F.col("category_name").isNull(), 1).otherwise(0)).alias("null_category"),
    F.sum(F.when(F.col("hash_watcher_id_norm") == "", 1).otherwise(0)).alias("empty_watcher_id"),
    F.sum(F.when(F.col("category_name_norm").isin(list(INVALID_CATEGORY_VALUES)), 1).otherwise(0)).alias("invalid_category_value"),
    F.sum(F.when(~F.col("category_name_norm").rlike("^[a-z].*"), 1).otherwise(0)).alias("category_not_starting_with_letter"),
).first()
print("[Preprocessing] Ringkasan baris bermasalah (kategori tidak eksklusif satu sama lain):")
print(f"  {drop_summary.asDict()}")

# --- Filter tunggal: gabungkan semua kondisi valid jadi satu pass ------------
df_clean = df_flagged.filter(
    F.col("hash_watcher_id").isNotNull() &
    F.col("category_name").isNotNull() &
    (F.col("hash_watcher_id_norm") != "") &
    (~F.col("category_name_norm").isin(list(INVALID_CATEGORY_VALUES))) &
    (F.col("category_name_norm").rlike("^[a-z].*")) &      # harus diawali huruf, buang kategori sampah/numerik
    (F.length(F.col("category_name_norm")) >= 3)            # buang string terlalu pendek untuk jadi genre valid
).withColumn("category_name", F.col("category_name_norm")) \
 .withColumn("hash_watcher_id", F.col("hash_watcher_id_norm")) \
 .drop("category_name_norm", "hash_watcher_id_norm")

# --- Isi missing string kolom lain (platform/os) menjadi "Unknown" ----------
string_cols_to_fill = [c for c in ["platform"] + os_col_candidates if c in df_clean.columns]
if string_cols_to_fill:
    df_clean = df_clean.fillna({c: "Unknown" for c in string_cols_to_fill})
    print(f"[Preprocessing] Isi missing string kolom {string_cols_to_fill} dengan 'Unknown'")

# --- Hilangkan duplicate pasangan (hash_watcher_id, category_name) ----------
df_clean = df_clean.dropDuplicates(["hash_watcher_id", "category_name"])

# --- Buang kategori long-tail: kategori yang dipakai < MIN_USER_PER_CATEGORY user --
# Kategori yang sangat jarang tidak bisa menghasilkan rule yang reliabel dan
# hanya menambah noise / memperlambat FPGrowth.
MIN_USER_PER_CATEGORY = 50

category_freq = df_clean.groupBy("category_name") \
    .agg(F.countDistinct("hash_watcher_id").alias("n_user")).cache()

n_category_before = category_freq.count()
valid_categories = category_freq.filter(F.col("n_user") >= MIN_USER_PER_CATEGORY)
n_category_after = valid_categories.count()
print(f"[Preprocessing] Kategori sebelum filter long-tail : {n_category_before:,}")
print(f"[Preprocessing] Kategori setelah filter (>= {MIN_USER_PER_CATEGORY} user) : {n_category_after:,}")

df_clean = df_clean.join(F.broadcast(valid_categories.select("category_name")), on="category_name", how="inner")

# --- Buang user outlier (kemungkinan bot: kategori ditonton jauh lebih banyak
#     dari mayoritas user, distorsi pola asosiasi) --------------------------
user_cat_count = df_clean.groupBy("hash_watcher_id") \
    .agg(F.countDistinct("category_name").alias("n_cat")).cache()

p99 = user_cat_count.approxQuantile("n_cat", [0.99], 0.01)[0]
outlier_users = user_cat_count.filter(F.col("n_cat") > p99).select("hash_watcher_id")
n_outlier_users = outlier_users.count()
print(f"[Preprocessing] Batas outlier jumlah kategori/user (P99) : {p99}")
print(f"[Preprocessing] User outlier dibuang                    : {n_outlier_users:,}")

df_clean = df_clean.join(outlier_users, on="hash_watcher_id", how="left_anti")
df_clean = df_clean.cache()

n_clean_final = df_clean.count()
print(f"\nJumlah baris sebelum preprocessing : {n_rows:,}")
print(f"Jumlah baris setelah preprocessing  : {n_clean_final:,}")
print(f"Persentase data tersisa             : {n_clean_final / n_rows:.2%}")

user_cat_count.unpersist()


# ------------------------------------------------------------------------------
# 6. TRANSFORMASI DATA: BASKET PER USER
# ------------------------------------------------------------------------------
print_header("TRANSFORMASI DATA (GROUPING PER USER)")

df_baskets = (
    df_clean.groupBy("hash_watcher_id")
    .agg(F.collect_set("category_name").alias("items"))
)

n_baskets_all = df_baskets.count()
print(f"Jumlah seluruh basket (user) hasil transformasi (sebelum filter): {n_baskets_all:,}")


# --- Buang kategori yang terlalu dominan (muncul di > 70% user) -------------
# Kategori semacam ini bertindak seperti "stop word": karena hampir semua
# user memilikinya, ia tidak informatif untuk membentuk rule rekomendasi
# yang bermakna dan justru mendominasi frequent itemsets.
DOMINANT_CATEGORY_THRESHOLD = 0.70

category_user_pct = category_freq.withColumn("pct", F.col("n_user") / F.lit(n_baskets_all))
dominant_categories_rows = category_user_pct.filter(F.col("pct") > DOMINANT_CATEGORY_THRESHOLD).collect()
dominant_categories = [r["category_name"] for r in dominant_categories_rows]
print(f"\n[Preprocessing] Kategori dominan (>{DOMINANT_CATEGORY_THRESHOLD:.0%} user), dibuang dari basket: {dominant_categories}")

if dominant_categories:
    df_baskets = df_baskets.withColumn(
        "items",
        F.array_except(F.col("items"), F.array([F.lit(c) for c in dominant_categories]))
    )

category_freq.unpersist()

# --- Filter basket minimal 2 item (perlu untuk rule antecedent -> consequent) --
df_baskets = df_baskets.filter(F.size(F.col("items")) >= 2)

# Repartition berdasarkan hash_watcher_id agar shuffle FPGrowth lebih seimbang;
# jumlah partisi diturunkan karena df_baskets jauh lebih kecil dari df_raw.
df_baskets = df_baskets.repartition(64, "hash_watcher_id").cache()
spark.conf.set("spark.sql.shuffle.partitions", "64")

n_baskets = df_baskets.count()
n_baskets_removed = n_baskets_all - n_baskets
print(f"\nJumlah basket setelah filter minimal 2 kategori : {n_baskets:,}")
print(f"Jumlah basket yang dihapus (item < 2 / dominan)  : {n_baskets_removed:,}")


# ------------------------------------------------------------------------------
# 7. TRAIN / TEST SPLIT (untuk evaluasi hit rate & coverage)
# ------------------------------------------------------------------------------
print_header("TRAIN / TEST SPLIT")

df_baskets_train, df_baskets_test = df_baskets.randomSplit([0.8, 0.2], seed=42)
df_baskets_train = df_baskets_train.cache()
df_baskets_train.count()
df_baskets_test = df_baskets_test.cache()

n_train = df_baskets_train.count()
n_test = df_baskets_test.count()
print(f"Jumlah basket training : {n_train:,}")
print(f"Jumlah basket testing  : {n_test:,}")


# ------------------------------------------------------------------------------
# 8. MODEL: FP-GROWTH (dilatih di TRAIN SET)
# ------------------------------------------------------------------------------
print_header("FP GROWTH")

MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.10

PARAM_CANDIDATES = [
    (MIN_SUPPORT, MIN_CONFIDENCE),
    (0.0005, 0.05),
    (0.0001, 0.05),
]
# Catatan: kandidat paling ekstrem (0.00005) di versi sebelumnya sengaja
# DIHAPUS karena pada skala 250 ribuan basket, itu berarti "frequent" hanya
# butuh ~13 user -- terlalu rentan noise untuk dianggap pola yang valid.

model = None
freq_itemsets = None
assoc_rules = None
n_freq_itemsets = 0
n_assoc_rules = 0
attempt_number = 0

for support_candidate, confidence_candidate in PARAM_CANDIDATES:
    attempt_number += 1
    MIN_SUPPORT = support_candidate
    MIN_CONFIDENCE = confidence_candidate

    print(f"\n--- Percobaan ke-{attempt_number} ---")
    print(f"  - minSupport    : {MIN_SUPPORT}")
    print(f"  - minConfidence : {MIN_CONFIDENCE}")

    fp_growth = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE)

    train_start = time.time()
    model = fp_growth.fit(df_baskets_train)
    train_duration = time.time() - train_start
    print(f"Training selesai dalam {train_duration:.2f} detik.")

    if freq_itemsets is not None:
        freq_itemsets.unpersist()
    if assoc_rules is not None:
        assoc_rules.unpersist()

    freq_itemsets = model.freqItemsets.cache()
    assoc_rules = model.associationRules.cache()

    n_freq_itemsets = freq_itemsets.count()
    n_assoc_rules = assoc_rules.count()

    print(f"Hasil percobaan ke-{attempt_number}: frequent itemsets = {n_freq_itemsets:,}, "
          f"association rules = {n_assoc_rules:,}")

    if n_assoc_rules > 0:
        print("Association rules berhasil ditemukan. Menghentikan percobaan lebih lanjut.")
        break
    else:
        print("Association rules masih kosong, mencoba parameter yang lebih longgar...")

print("\nParameter FP-Growth final yang digunakan:")
print(f"  - Jumlah percobaan    : {attempt_number}")
print(f"  - minSupport final    : {MIN_SUPPORT}")
print(f"  - minConfidence final : {MIN_CONFIDENCE}")


# ------------------------------------------------------------------------------
# 9. FILTER KUALITAS RULE (baru: fokus pada kebermaknaan, bukan kuantitas)
# ------------------------------------------------------------------------------
print_header("FILTER KUALITAS ASSOCIATION RULES")

MIN_LIFT = 1.2                  # rule harus menunjukkan asosiasi positif yang jelas, bukan mendekati acak (lift=1)
MIN_RULE_SUPPORT_COUNT = 30      # jumlah transaksi absolut minimum, agar rule tidak berbasis segelintir user

assoc_rules_quality = (
    assoc_rules
    .withColumn("support_count", (F.col("support") * F.lit(n_train)).cast("long"))
    .filter((F.col("lift") >= MIN_LIFT) & (F.col("support_count") >= MIN_RULE_SUPPORT_COUNT))
    .cache()
)

n_assoc_rules_quality = assoc_rules_quality.count()
print(f"Rule mentah dari FPGrowth                         : {n_assoc_rules:,}")
print(f"Rule setelah filter lift >= {MIN_LIFT} & support_count >= {MIN_RULE_SUPPORT_COUNT} : {n_assoc_rules_quality:,}")


# ------------------------------------------------------------------------------
# 9.5. REKOMENDASI CROSS-GENRE
# ------------------------------------------------------------------------------
print_header("REKOMENDASI CROSS-GENRE")

top_rules = (
    assoc_rules_quality
    .orderBy(
        F.desc("lift"),
        F.desc("confidence"),
        F.desc("support_count")
    )
    .select(
        "antecedent",
        "consequent",
        "confidence",
        "lift",
        "support_count"
    )
    .limit(10)
    .collect()
)

if len(top_rules) == 0:
    print("Tidak ada rekomendasi cross-genre yang memenuhi kriteria.")
else:
    print("Top 10 rekomendasi cross-genre berdasarkan Association Rules:\n")

    for i, rule in enumerate(top_rules, start=1):
        antecedent = ", ".join(rule["antecedent"])
        consequent = ", ".join(rule["consequent"])

        print(f"{i}. User yang menonton [{antecedent}]")
        print(f"   → Direkomendasikan menonton [{consequent}]")
        print(f"      Confidence : {rule['confidence']:.4f}")
        print(f"      Lift       : {rule['lift']:.4f}")
        print(f"      Support    : {rule['support_count']}")
        print("-" * 70)


# ------------------------------------------------------------------------------
# 10. EVALUASI: HIT RATE & COVERAGE (leave-one-out di TEST SET)
# ------------------------------------------------------------------------------
print_header("EVALUASI: HIT RATE & COVERAGE")

# Sembunyikan 1 item terakhir tiap basket test sebagai "ground truth" yang
# harus ditebak oleh rule hasil training (skema leave-one-out sederhana).
split_schema = StructType([
    StructField("history", ArrayType(StringType())),
    StructField("held_out", StringType()),
])

def split_last_item(items):
    items_sorted = sorted(items)  # urutan tetap agar hasil reproducible
    return (items_sorted[:-1], items_sorted[-1])

split_udf = F.udf(split_last_item, split_schema)

df_test_split = (
    df_baskets_test
    .withColumn("split", split_udf(F.col("items")))
    .select(
        "hash_watcher_id",
        F.col("split.history").alias("items"),
        F.col("split.held_out").alias("held_out"),
    )
)

df_test_pred = model.transform(df_test_split).cache()

n_test_total = df_test_split.count()
n_test_with_pred = df_test_pred.filter(F.size(F.col("prediction")) > 0).count()
n_hit = df_test_pred.filter(F.array_contains(F.col("prediction"), F.col("held_out"))).count()

coverage = n_test_with_pred / n_test_total if n_test_total > 0 else 0.0
hit_rate = n_hit / n_test_total if n_test_total > 0 else 0.0

print(f"Jumlah basket test                                : {n_test_total:,}")
print(f"User yang mendapat rekomendasi non-kosong (coverage): {n_test_with_pred:,} ({coverage:.2%})")
print(f"User yang item held-out-nya berhasil ditebak (hit)   : {n_hit:,} ({hit_rate:.2%})")
print("\nInterpretasi:")
print("  - Coverage tinggi -> model bisa memberi rekomendasi untuk sebagian besar user.")
print("  - Hit rate tinggi -> rekomendasi yang diberikan relevan dengan perilaku nyata user,")
print("    bukan hanya banyak secara jumlah.")



# ------------------------------------------------------------------------------
# 11. ANALISIS HASIL
# ------------------------------------------------------------------------------
print_header("ANALISIS HASIL")

avg_items_per_basket = df_baskets.select(F.avg(F.size(F.col("items"))).alias("avg_items")).first()["avg_items"]

print(f"Jumlah transaksi (basket) sebelum filter   : {n_baskets_all:,}")
print(f"Jumlah basket dipakai (>= 2 item)           : {n_baskets:,}")
print(f"  - Training                                : {n_train:,}")
print(f"  - Testing                                 : {n_test:,}")
print(f"Rata-rata jumlah item per basket             : {avg_items_per_basket:.4f}")
print(f"Frequent itemsets (mentah)                   : {n_freq_itemsets:,}")
print(f"Association rules (mentah)                   : {n_assoc_rules:,}")
print(f"Association rules (setelah filter kualitas)  : {n_assoc_rules_quality:,}")
print(f"Coverage (evaluasi test set)                 : {coverage:.2%}")
print(f"Hit rate (evaluasi test set)                 : {hit_rate:.2%}")
print(f"Parameter final (minSupport, minConfidence)  : ({MIN_SUPPORT}, {MIN_CONFIDENCE})")

if n_assoc_rules_quality > 0:
    top_lift_rule = (
        assoc_rules_quality.orderBy(F.desc("lift"))
        .select("antecedent", "consequent", "confidence", "lift", "support", "support_count")
        .first()
    )
    print("\nRule berkualitas dengan lift tertinggi:")
    print(f"  Antecedent    : {top_lift_rule['antecedent']}")
    print(f"  Consequent    : {top_lift_rule['consequent']}")
    print(f"  Confidence    : {top_lift_rule['confidence']:.4f}")
    print(f"  Lift          : {top_lift_rule['lift']:.4f}")
    print(f"  Support       : {top_lift_rule['support']:.4f}")
    print(f"  Support count : {top_lift_rule['support_count']:,}")
else:
    print("\nTidak ada rule berkualitas yang ditemukan. Kemungkinan penyebab:")
    print("  1. MIN_LIFT / MIN_RULE_SUPPORT_COUNT terlalu ketat untuk skala data ini.")
    print("  2. MIN_USER_PER_CATEGORY atau DOMINANT_CATEGORY_THRESHOLD membuang terlalu")
    print("     banyak kategori sehingga kombinasi antar-kategori menjadi sedikit.")
    print("  3. Perilaku menonton user memang cenderung acak/tidak berpola kuat.")


# ------------------------------------------------------------------------------
# 12. EVALUASI METRIK (penjelasan)
# ------------------------------------------------------------------------------
print_header("EVALUASI METRIK")
print("""
1. SUPPORT       : Support(A) = jumlah_transaksi_mengandung_A / total_transaksi
2. CONFIDENCE    : Confidence(A -> B) = Support(A dan B) / Support(A)
3. LIFT          : Lift(A -> B) = Confidence(A -> B) / Support(B)
                   Lift > 1 = asosiasi positif, Lift = 1 = independen, Lift < 1 = negatif
4. HIT RATE      : proporsi user di test set yang item tersembunyinya berhasil
                   ditebak oleh rule hasil training (relevansi rekomendasi)
5. COVERAGE      : proporsi user di test set yang mendapat rekomendasi non-kosong
                   (seberapa luas model dapat memberi saran)

Rule dengan lift tinggi + support_count memadai menunjukkan pasangan genre yang
benar-benar saling berkaitan secara kebiasaan menonton, sementara hit rate &
coverage memberi bukti kuantitatif bahwa pola tersebut juga berguna untuk
memprediksi perilaku user yang belum pernah dilihat model (data test).
""")


# ------------------------------------------------------------------------------
# 13. WAKTU EKSEKUSI
# ------------------------------------------------------------------------------
print_header("WAKTU EKSEKUSI")
total_duration = elapsed_since_start()
minutes = int(total_duration // 60)
seconds = total_duration % 60
print(f"Total waktu eksekusi program: {total_duration:.2f} detik ({minutes} menit {seconds:.2f} detik)")


# ------------------------------------------------------------------------------
# 14. BERSIHKAN CACHE & TUTUP SPARK SESSION
# ------------------------------------------------------------------------------
df_clean.unpersist()
df_baskets.unpersist()
df_baskets_train.unpersist()
df_baskets_test.unpersist()
df_test_pred.unpersist()
freq_itemsets.unpersist()
assoc_rules.unpersist()
assoc_rules_quality.unpersist()

print("\nProgram selesai. Menutup SparkSession...")
spark.catalog.clearCache()

time.sleep(3)

spark.stop()
print("SparkSession ditutup. Selesai.")