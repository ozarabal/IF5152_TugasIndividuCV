
# IF5152 — Tugas Individu Computer Vision (Minggu 3–6)
# Rafif Ardhinto Ichwantoro
# 13522159

Repositori ini berisi aplikasi **modular** dan **runnable** yang mengintegrasikan:
1) *Image Filtering*, 2) *Edge Detection & Sampling*, 3) *Feature/Interest Points*, dan 4) *Camera Geometry & Calibration*.

## Struktur Folder
```
Nama_NIM_IF5152_TugasIndividuCV/
├── 01_filtering/
│   └── run_filtering.py          # Gaussian & Median; simpan before/after + tabel parameter
├── 02_edge/
│   └── run_edge.py               # Sobel & Canny; variasi threshold + sampling; tabel parameter
├── 03_featurepoints/
│   └── run_featurepoints.py      # Harris & ORB; marking + statistik
├── 04_geometry/
│   └── run_geometry.py           # Proyeksi/projective & affine transform; matriks parameter
├── utils/
│   └── common.py                 # Helper: load image, save, csv logger, konversi gray
└── README.md
```

Seluruh script menulis hasil ke subfolder `out/` di masing‑masing modul: **gambar _before/after_**, **overlay/edge map**, serta **CSV parameter/statistik**.

## Cara Menjalankan (Environment minimal)
- Python ≥ 3.9
- Paket: `numpy`, `matplotlib`, `scikit-image`

Contoh instalasi cepat:
```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:    .venv\Scripts\activate
pip install numpy matplotlib scikit-image
```

Jalankan tiap modul (bebas urutan):
```bash
python 01_filtering/run_filtering.py
python 02_edge/run_edge.py
python 03_featurepoints/run_featurepoints.py
python 04_geometry/run_geometry.py
```

## Workflow / Pipeline
1. **Load Data** — Memuat empat citra standar dari `skimage.data`: `camera`, `coins`, `checkerboard`, `astronaut`. Semua diproses sebagai float [0..1], grayscale bila perlu.
2. **Filtering** — Mengurangi noise & mengontrol smoothing:
   - *Gaussian* (σ ∈ {0.8, 2.0}) untuk blur halus/anti‑noise.
   - *Median* (radius ∈ {1, 3}) untuk meredam salt‑&‑pepper.
   - **Output**: gambar *before/after* + `filter_params.csv` (nama gambar, jenis filter, parameter).
3. **Edge Detection & Sampling** — Mengekstrak tepi dan mengamati efek skala/threshold:
   - *Sobel* (gradien) dan *Canny* (σ=1.2, variasi *low/high threshold*).
   - *Sampling/rescale* s ∈ {1.0, 0.5} untuk melihat efek resolusi pada tepi.
   - **Output**: peta tepi & `edge_params.csv` berisi parameter dan skala sampling.
4. **Feature/Interest Points** — Menemukan sudut/kunci lokal:
   - *Harris* (k=0.05, σ=1.2) dengan `corner_peaks` → titik ditandai lingkaran merah.
   - *ORB* (n_keypoints=300) → keypoint cepat + deskriptor.
   - **Output**: citra bertanda & `feature_stats.csv` (jumlah fitur & parameter).
5. **Geometry & Calibration (Simulasi)** — Mendemonstrasikan proyeksi kamera sederhana:
   - Estimasi **Projective Transform (Homography)** dari korespondensi sudut gambar ke bidang miring → hasil *warped* checkerboard.
   - Perbandingan **Affine Transform** (scale+shear+translate).
   - **Output**: overlay/hasil transformasi & `transform_params.csv` berisi matriks 3×3 (projective) dan matriks affine.

## Fitur Unik
- *RunLog* ringan untuk konsistensi tabel parameter/statistik.
- Gambar disimpan sebagai PNG 8‑bit sehingga portabel untuk penilaian.
- Semua modul dapat dijalankan independen maupun dipakai ulang sebagai library.
