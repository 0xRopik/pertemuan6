# Pertemuan 6 — Random Forest untuk Klasifikasi 🎓

## Tujuan
Melatih dan mengevaluasi model Random Forest untuk memprediksi kelulusan mahasiswa.

---

## Langkah Cepat Menjalankan
1️⃣ Pastikan file `processed_kelulusan.csv` ada di folder ini.  
2️⃣ Jalankan notebook:
```
jupyter notebook model_rf_kelulusan.ipynb
```
atau jalankan training script:
```
python train_rf.py
```
Hasil Output
```
rf_model.pkl → model tersimpan
roc_test.png → grafik ROC
pr_test.png → grafik Precision-Recall
```
Testing Model
Gunakan Flask API:
```
python app.py
```
Lalu kirim data:
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
-d '{"IPK": 3.4, "Jumlah_Absensi": 4, "Waktu_Belajar_Jam": 7, "Rasio_Absensi": 0.3, "IPK_x_Study": 23.8}'
```
