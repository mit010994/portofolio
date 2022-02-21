Link repository github M. Iqbal Tawakkal: https://github.com/mit010994/proyek

Role:
Sebagai tim data science dari suatu perusahaan e-commerce yang bernama Grow-Ject, kami diminta untuk memberikan solusi bisnis untuk mengatasi masalah bisnis yang ada dengan tujuan meningkatkan performa bisnis berdasarkan data yang tersedia

Problem Statement:
Sebuah studi mengatakan bahwa 69% customer kemungkinan tidak kembali lagi jika barang telat sampai. 
~60% product shipment yang dikirim oleh Grow-Ject merupakan late delivery. 
Bisnis prihatin bahwa late delivery rate yang cenderung tinggi mempengaruhi customer retention rate.

Latar belakang masalah:
- 69% of consumers “are much less or less likely to shop with a retailer in the future if an item they purchased is not delivered within two days of the date promised.”
- 17% of respondents will stop shopping with a retailer after receiving a late delivery one time. 
- 55% of respondents will stop shopping with a retailer after receiving a late delivery two to three times.

Goals:
Meningkatkan customer retention rate

Objectives: 
Membuat model machine learning yang bertujuan untuk memprediksi late delivery
Menggali data untuk mengetahui penyebab late delivery
Memberikan treatment yang tepat kepada customer yang shipmentnya diprediksi sebagai late delivery untuk me-retain customer (sebagai bagian dari risk control dalam risk management)

Business Metrics:
Customer retention rate

Hasil EDA:
- Tidak ada kolom yang null dan duplicate
- Kolom customer_care_calls dan customer_rating terlihat distribusinya cukup simetrik (Mean dan Median tidak berbeda jauh)
- Kolom prior_purchases dan discount_offered terlihat Positively Skewed
- Kolom late_delivery bernilai boolean
- Distribusi customer_care_calls: positively skewed (mild)
- Distribusi customer_rating: Uniform
- Distribusi cost_of_product: Bimodal
- Distribusi prior_purchases: Positively Skewed
- Distrbusi discount_offered: Positively Skewed (severe)
- Distribusi weight_in_grams: U-Shaped / Bimodal
- Outlier paling banyak ditemukan pada parameter Discount_offered, sedangkan pada fitur Prior_purchases memiliki beberapa outlier yang jelas terlihat
- Kebanyakan pengiriman dilakukan dari gudang F
- Metode pengiriman di dominasi oleh pengiriman menggunakan Ship
- Kebanyakan barang yang dikirim itu prioritasnya Low

Hasil multivariate analysis:
- discount_offered` memiliki korelasi positif dengan target `late delivery` yaitu sebesar 0.40
- weight_in_gms` memiliki korelasi negatif dengan target `late delivery` yaitu sebesar -0.27
- Shipment dengan `cost_of_the_product` pada sekitar 150 cenderung untuk late dibanding on time
- Tidak terlihat bahwa ada antar feature yang memiliki korelasi yang kuat

Data preprocessing:
- Pengecekan outlier menggunakan Z-score dan membuang Z-score yang lebih dari 3
- Class imbalance handling: 40 on-time : 60 late delivery → oversampling SMOTE (frac=0.75)
- Splitting data menjadi train set dan test set (80:20)
- Transformasi distribusi (np.sqrt untuk weight in gms, prior purchases, discount offered, dan cost of the product)
- Feature scaling (MinMaxScaler)
- Label encoding untuk gender (M = 1 & F = 0)
- OneHotEncoding untuk warehouse_block, mode_of_shipment, product_importance
- Step 5, 6, dan 7 dilakukan kepada test set juga

Feature selection:
- Chi Squared test: melihat predictive power dari categorical features. Ternyata hanya product_importance_high yang memiliki score tinggi
- Feature importances menggunakan RandomForestClassifier
- Feature Rankings menggunakan Boruta dari hasil feature importances RandomForestClassifier, dan diambil fitur ranking 1 dan 2 saja 
- Multicollinearity: Pearson correlation matrix untuk melihat fitur yang redundant, dibuang yang secara feature importance lebih rendah
- Melalui tahapan-tahapan feature selection, berikut adalah list fitur yang terseleksi untuk digunakan pada machine learning model:
1. cost_of_the_product
2. discount_offered
3. trf_weight_in_gms (np.sqrt)
4. product_importance_high (satu-satunya categorical feature yang memiliki chi-squared statistic tinggi)

Machine Learning Modelling:
- Evaluation metric yang digunakan: Recall
- Hasil model selection menggunakan lazypredict: tree-based, bagging, stacking, dan boosting models
- Metrics untuk production grade yang ditargetkan:
1. Recall untuk late delivery : 0.75
2. Recall untuk on-time delivery : 0.5
- Cross validation dan prediction awal diuji menggunakan metrics yaitu averaged-micro recall karena pada prosesnya scoring tidak bisa dipilih untuk dua class secara terpisah
- Hasilnya, XGBoostRF merupakan model yang paling unggul (dan paling viable untuk hyperparameter tuning)
- Recall sebelum hyperparameter tuning:
1. Late: 0.49
2. On-time: 0.97
- Recall setelah hyperparameter tuning:
1. Late: 0.75
2. On-time: 0.53
- Hyperparameter yang digunakan:
1. 'booster': 'dart',
2. 'eta': 0.025,
3. 'gamma': 0,
4. 'max_delta_step': 3,
5. 'max_depth': 9,
6. 'min_child_weight': 0.8,
7. 'reg_alpha': 2,
8. 'reg_lambda': 0.1,
9. 'tree_method': 'hist',
10. 'scale_pos_weight': 1.4,
11. 'subsample': 0.7,
12. 'n_jobs': -1,
13. 'random_state': 42,

Business recommendation:
Dengan model machine learning ini, perusahaan dapat memperkirakan apakah barang yang dikirim akan terlambat atau tepat waktu. 
Barang yang diprediksi akan terlambat akan dimasukkan ke dalam prioritized tracking, yang mana barang tersebut akan lebih dimonitor daripada barang yang diprediksi akan tepat waktu.
Pemilik barang yang terlambat datang akan menerima pesan/notifikasi permintaan maaf dari perusahaan serta info mengenai penyebab keterlambatan tersebut.
Apabila barang masih tidak tiba hingga waktu toleransi yang telah ditentukan, pemilik barang tersebut akan mendapatkan kompensasi berupa kupon potongan harga (diskon) atau refund
