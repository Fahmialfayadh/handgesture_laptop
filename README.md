# Hand Gesture Laptop Controller V2.0

Aplikasi pengontrol kursor mouse laptop secara *real-time* berbasis gestur tangan menggunakan **MediaPipe Hands** dan **Machine Learning (K-Nearest Neighbors / Decision Tree / Logistic Regression)**.

Dirancang khusus agar ringan di CPU (30–60 FPS), bebas *jitter*, mendukung multi-platform (**Linux & Windows**), dan bisa menggunakan **Webcam Laptop** maupun **Kamera HP (via IP lokal/WiFi)**.

---

## 1. Pemetaan Gestur Tangan (1 Gestur = 1 Fungsi)

| Label | Gestur Tangan | Mode / Aksi | Deskripsi |
| :---: | :--- | :---: | :--- |
| `0` | **Telapak Tangan Terbuka** | `STANDBY` | Kursor diam di tempat, tidak memicu klik/scroll. |
| `1` | **Hanya Jari Telunjuk/Satu Jari** | `MOVE` | Menggerakkan kursor di layar. Dilengkapi perataan adaptif (presisi saat pelan, gesit saat cepat). |
| `2` | **Telunjuk + Jari Tengah/ Dua Jari** | `SCROLL` | Scroll halaman kontinu (gerakkan tangan ke atas / bawah dari posisi awal). |
| `3` | **Pose 3 Jari** | `LEFT_CLICK` | Memicu 1 kali klik kiri mouse (stabil dan bebas salah picu). |
| `4` | **Pose 4 Jari** | `RIGHT_CLICK` | Memicu 1 kali klik kanan mouse (membuka menu konteks). |

---

## 2. Struktur File Proyek

```text
├── src/                       # Modul inti aplikasi kontroler
│   ├── __init__.py            # Inisialisasi package & export
│   ├── smoother.py            # GestureSmoother: stabilisasi probabilitas EMA & anti-flicker
│   ├── actions.py             # ActionEngine: eksekusi kursor, scroll kontinu, & klik
│   └── utils.py               # Fungsi normalisasi fitur 42 koordinat & mapping layar
├── gesture_controller.py      # Entry point utama aplikasi kontroler real-time
├── best_gesture_model.joblib  # Model ML terbaik yang sudah dilatih (KNN k=5, Test Acc 99.84%)
├── gesture_training.ipynb     # Pipeline lengkap: akuisisi data, benchmarking model, & ekspor
├── dataset_gestures.csv       # Dataset koordinat sendi tangan (50.006 data terstandarisasi)
├── requirements.txt           # Daftar dependensi pustaka Python
├── run.sh                     # Runner praktis untuk Linux / macOS
└── run.bat                    # Runner praktis untuk Windows
```

---

## 3. Cara Instalasi & Menjalankan

### A. Persiapan Lingkungan (Environment)

**Menggunakan `uv` (Direkomendasikan di Linux):**
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Menggunakan `pip` biasa (Windows / Linux):**
```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

---

### B. Menjalankan Aplikasi

#### 1. Menggunakan Webcam Bawaan Laptop:
* **Linux:**
  ```bash
  ./run.sh
  # atau
  python gesture_controller.py
  ```
* **Windows:**
  Klik dua kali file `run.bat` atau jalankan di terminal:
  ```cmd
  run.bat
  ```

#### 2. Menggunakan Kamera HP via IP Lokal (WiFi):
1. Install aplikasi seperti **IP Webcam** (Android) di HP.
2. Pastikan HP dan Laptop terhubung ke WiFi yang sama (atau via Hotspot HP).
3. Buka aplikasi di HP dan pilih **Start Server** (akan muncul URL di layar HP, misal `http://192.168.1.15:8080`).
4. Jalankan script dengan menyertakan URL stream:
   ```bash
   ./run.sh http://192.168.1.15:8080/video
   ```

*Untuk keluar dari aplikasi, tekan tombol **`q`** atau **`Esc`** pada jendela kamera.*

---

## 4. Arsitektur Teknis & Logika Kode

Aplikasi pada `gesture_controller.py` menggunakan arsitektur modular yang memisahkan antara *Vision*, *Filtering/Stabilization*, dan *Action*:

1. **Feature Extractor (MediaPipe Hands):**
   * Mengekstrak 21 titik koordinat anatomis tangan manusia.
   * Fungsi `extract_features()` mentranslasikan koordinat ke pergelangan tangan (*wrist* titik 0) dan menormalisasi skala jarak telapak tangan (titik 9) agar kebal terhadap jarak dekat/jauh dari kamera.
2. **GestureSmoother (Stabilisasi Probabilitas):**
   * Menggunakan Exponential Moving Average (EMA) pada `predict_proba()` model ML.
   * Mencegah *flickering* (pergantian mode mendadak antar-frame akibat sensor noise). Mode hanya berpindah jika probabilitas $\ge 60\%$ dan stabil selama minimal 4 frame berturut-turut.
3. **ActionEngine (Eksekusi Kursor):**
   * **Adaptive Cursor Smoothing:** Menyesuaikan nilai $\alpha$ perataan secara dinamis berdasarkan kecepatan gerak tangan.
   * **Anchor-based Scroll:** Menetapkan titik jangkar saat masuk mode `SCROLL`, dilengkapi zona netral $\pm 8\text{ px}$ agar scroll halus dan tidak meloncat.
   * **Single-Fire Click:** Mencegah spam klik dengan proteksi *debounce timer*.

---

## 5. Hasil Evaluasi & Komparasi Model

Berdasarkan pengujian pada 50.006 sampel (42 fitur spasial normalisasi) dengan rasio *train-test split* 80:20:

| Model | Train Accuracy | Test Accuracy | Latensi Inferensi | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **K-Nearest Neighbors ($k=5$)** | **99.88%** | **99.84%** | **~5.02 ms** | **Dipilih (Akurasi tertinggi, gap latih-uji 0.04%)** |
| Logistic Regression | 98.77% | 98.69% | ~0.34 ms | Akurasi lebih rendah dibanding KNN |
| Decision Tree (max_depth=6) | 98.12% | 97.77% | ~0.30 ms | Akurasi terendah di antara ketiga model |

Model **KNN** dipilih untuk *deployment* (`best_gesture_model.joblib`) karena memberikan akurasi generalisasi terbaik tanpa *overfitting*, dengan latensi inferensi ~5 ms yang masih jauh di bawah batas waktu per frame (16.67 ms untuk 60 FPS / 33.3 ms untuk 30 FPS).

---

## 6. Pelatihan Ulang Model (Opsional)

Jika ingin menambah variasi gestur atau melatih model baru:
1. Buka notebook [gesture_training.ipynb](file:///home/data/kuliah/rka-knowledge/courses/LBE_KCV/draft_fp_laptopcontroller/gesture_training.ipynb).
2. Jalankan **Cell 6** untuk merekam data koordinat baru langsung melalui webcam *(tahan pose tangan dan tekan tombol 0 s.d. 4 pada keyboard)*.
3. Jalankan sel-sel berikutnya untuk melatih, mengevaluasi ketiga model klasifikasi, dan mengekspor model terbaik ke `best_gesture_model.joblib`.

