# PPEye Vision: SOP Compliance Monitor

*[Access Link Here](https://papieye.streamlit.app/)*

## Repository Outline

Berikut adalah gambaran singkat dari file-file yang ada di repository.

1. **`FP_03_EDA.ipynb`**Notebook ini digunakan untuk melakukan *Exploratory Data Analysis (EDA)* terhadap dataset proyek. Di dalamnya dilakukan analisis awal seperti distribusi kelas, visualisasi data, serta identifikasi potensi masalah dalam data.
2. **`FP_03_Inference_Camera.py`**Script Python yang digunakan untuk melakukan inferensi secara real-time menggunakan kamera. File ini memanfaatkan model YOLOv8 untuk mendeteksi penggunaan APD secara langsung dari video kamera.
3. **`FP_03_Inference.ipynb`**Notebook ini menjalankan proses inferensi dari model terhadap gambar atau video. Berisi kode untuk memuat model YOLOv8 dan menjalankan prediksi terhadap data input yang tidak real-time.
4. **`FP_03_Model.ipynb`**Notebook ini digunakan untuk proses pelatihan model YOLOv8. Termasuk di dalamnya adalah konfigurasi data, augmentasi, training loop, serta penyimpanan model hasil training.
5. **`FP_03_Splitting_Data.ipynb`**Notebook ini bertugas untuk memisahkan data menjadi folder `train`, `valid`, dan `test`. Proses ini penting untuk persiapan pelatihan model yang membutuhkan struktur data terpisah.
6. **`data.yaml`**File konfigurasi yang digunakan oleh YOLOv8 untuk mengetahui label serta jalur direktori dataset yang digunakan dalam proses pelatihan dan inferensi.
7. **`yolov8n.pt`**File model terlatih (YOLOv8 Nano) yang digunakan untuk inferensi. File ini dihasilkan setelah proses training dan menjadi model utama untuk deteksi APD.
8. **`README.md`**File dokumentasi umum yang menjelaskan tujuan proyek, cara menjalankan file, serta penjelasan ringkas tentang setiap bagian dalam repositori.
9. **`dataset/`**Folder ini berisi dataset gambar dan anotasi yang digunakan untuk pelatihan model YOLOv8.
10. **`runs/detect/train/`**Folder output dari proses pelatihan YOLOv8. Berisi hasil pelatihan seperti grafik loss dan metrik evaluasi (`results.png`), konfigurasi model (`opt.yaml`), dan bobot model hasil pelatihan (`weights/best.pt`, `weights/last.pt`). Folder ini dihasilkan secara otomatis oleh YOLOv8 saat training dijalankan.
11. **args.yaml**
    Konfigurasi parameter saat training seperti epoch, batch size, dan dataset path.
12. **best.pt**Model hasil pelatihan terbaik.
13. **last.pt**Model hasil pelatihan terakhir.
14. **BoxF1_curve.png**Grafik kurva F1 Score per epoch.
15. **BoxP_curve.png**Grafik kurva Precision per epoch.
16. **BoxPR_curve.png**Grafik Precision-Recall curve.
17. **BoxR_curve.png**Grafik kurva Recall per epoch.
18. **confusion_matrix_normalized.png**Confusion matrix hasil validasi (dalam bentuk persentase).
19. **confusion_matrix.png**Confusion matrix hasil validasi (dalam jumlah absolut).
20. **labels_correlogram.jpg**Visualisasi hubungan antar label.
21. **labels.jpg**Visualisasi distribusi label dalam dataset.
22. **losskomponen.png**Grafik loss training (box loss, obj loss, cls loss).
23. **results.csv**File CSV berisi metrik evaluasi per epoch.
24. **results.png**Ringkasan grafik metrik pelatihan selama training.
25. **train_batch0.jpg**Visualisasi batch pertama dari data pelatihan beserta hasil prediksi model.
26. **train_batch1.jpg**Visualisasi batch kedua dari data pelatihan dan prediksi model.
27. **train_batch2.jpg**Visualisasi batch ketiga dari data pelatihan dan prediksi model.
28. **train_batch3320.jpg**Visualisasi batch ke-3320 dari data pelatihan untuk monitoring model.
29. **train_batch3321.jpg**Visualisasi batch ke-3321 dari data pelatihan.
30. **train_batch3322.jpg**Visualisasi batch ke-3322 dari data pelatihan.
31. **val_batch0_labels.jpg**Gambar validasi batch pertama yang menampilkan label ground truth.
32. **val_batch0_pred.jpg**Gambar validasi batch pertama yang menampilkan hasil prediksi model.
33. **val_batch1_labels.jpg**Gambar validasi batch kedua yang menampilkan label ground truth.
34. **val_batch1_pred.jpg**Gambar validasi batch kedua yang menampilkan hasil prediksi model.
35. **val_batch2_labels.jpg**Gambar validasi batch ketiga yang menampilkan label ground truth.
36. **val_batch2_pred.jpg**
    Gambar validasi batch ketiga yang menampilkan hasil prediksi model.

## Problem Background

PT Mayora Indah Tbk menghadapi tantangan dalam memastikan kepatuhan penggunaan APD di area produksi sesuai standar GMP dan ISO 9001. Pengawasan manual sering kali tidak efektif dan berisiko terhadap keamanan produk. Melalui proyek ini, Mayora mengembangkan PPEye, sistem berbasis AI untuk mendeteksi kelengkapan APD secara otomatis menggunakan teknologi object detection. Solusi ini membantu meningkatkan kepatuhan, mengurangi kesalahan manusia, dan menjaga standar keamanan produk di tengah ketatnya persaingan industri FMCG.

## Project Output

Project disini menghasilkan sebuah model Object Detection untuk mengecek kestandaran dari pekerja yang ada di pabrik supaya sesuai dengan SOP yang ada. Dengan menggunakan hairnet, apron, mask, dan gloves. Dimana nanti alat ini akan ditaruh didepapn pintu depan disaat pekerja masuk ke ruang produksi setelah mereka menggukan pakaian yang sesuai denga SOP perusahaan.

### Analisis Dataset: Global Music Streaming Trends and Listener Insights

**Sumber Data**:
[https://universe.roboflow.com/personcountingsonu/sonu_person-20may](https://universe.roboflow.com/personcountingsonu/sonu_person-20may) dengan menambahkan dataset real dan mencari dari sumber sumber lainnya sehingga membantu model PPEye dapat belajar dengan baik

### **Karakteristik Data**:

1. **Dimensi Data**:

   - 8,032 foto dari berbagai resolusi dan sumber.
2. **Anotasi label yang dipakai**:

   - **Dipakai** (4 jenis):
     * **Gloves** : Menunjukkan karyawan mengenakan sarung tangan.
     * **Hairnet** : Menunjukkan karyawan mengenakan penutup kepala.
     * **Mask** : Menunjukkan karyawan mengenakan masker.
     * **Apron** : Menunjukkan karyawan mengenakan celemek.
   - **Tidak dipakai** (4 jenis):
     * **No gloves** : Menunjukkan karyawan tidak mengenakan sarung tangan.
     * **No hairnet** : Menunjukkan karyawan tidak mengenakan penutup kepala.
     * **No mask** : Menunjukkan karyawan tidak mengenakan masker.
     * **No apron** : Menunjukkan karyawan tidak mengenakan celemek.
3. **File rusak dan missing annotations**: Dari dataset ini tidak ada file yang rusak dan missing annotations
4. **Sebaran Resolusi Dataset**: Tinggi gambar bervariasi dari 480 picel hingga 899 pixel. Untuk lebar gambar berkisar antara 596 pixel hingga 1.032 pixel.

## Method

Di proyek ini, kami mengembangkan pipeline untuk membangun model deteksi objek yang bertujuan memeriksa kepatuhan pekerja terhadap Standar Operasional Prosedur (SOP) di pabrik, khususnya pada penggunaan peralatan pelindung diri (PPE) seperti hairnet, apron, mask, dan gloves. Alat ini dirancang untuk ditempatkan di pintu masuk ruang produksi, memastikan pekerja telah mengenakan pakaian sesuai SOP sebelum memulai aktivitas. Proses dimulai dengan pengumpulan dataset yang beragam: kami memanfaatkan dataset awal dari situs web, dilengkapi dengan data tambahan dari berbagai sumber, termasuk gambar nyata serta foto asli yang kami ambil sendiri, untuk menciptakan dataset yang representatif dan relevan.

Sebelum digunakan untuk pelatihan model, data divalidasi untuk memastikan kualitas, dengan fokus pada anotasi yang akurat dan konsisten. Model deteksi objek dikembangkan menggunakan pendekatan visi komputer, dengan implementasi berbasis YOLO untuk mengidentifikasi dan mengklasifikasikan penggunaan PPE secara real-time. Hasilnya, sistem ini mampu memberikan umpan balik instan mengenai kepatuhan SOP, yang dapat dimanfaatkan oleh tim manajemen pabrik untuk meningkatkan keselamatan kerja dan mematuhi regulasi industri pangan.

## Stacks

* **Python** : Digunakan untuk scripting, pengolahan data, dan validasi data dalam proyek.
* **NumPy** : Digunakan untuk komputasi numerik dan manipulasi array data.
* **Pandas** : Digunakan untuk manipulasi, transformasi, dan analisis data dalam bentuk tabel.
* **OpenCV (cv2)** : Digunakan untuk pemrosesan dan analisis gambar, termasuk pembacaan serta preprocessing data visual.
* **Matplotlib** : Digunakan untuk visualisasi data dan plot statistik.
* **Torch** : Digunakan untuk pembangunan dan pelatihan model deep learning, khususnya untuk deteksi objek.
* **Collections (Counter, defaultdict)** : Digunakan untuk operasi data terstruktur seperti penghitungan frekuensi dan penyimpanan data default.
* **SciPy (gaussian_kde)** : Digunakan untuk analisis statistik lanjutan, termasuk estimasi kepadatan kernel.
* **tqdm** : Digunakan untuk menampilkan progress bar selama iterasi proses panjang.
* **Ultralytics (YOLO)** : Digunakan untuk implementasi model YOLO guna deteksi objek berbasis visi komputer.
* **IPython.display (Image, display)** : Digunakan untuk menampilkan gambar dan visualisasi langsung dalam lingkungan Jupyter atau notebook.

## Reference

* [Deep learning as A Monitoring Measure for Good Hygiene Practices in Food Factories](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22111NTOU0252004%22.&searchmode=basic&extralimit=asc=%22%E5%9C%8B%E7%AB%8B%E8%87%BA%E7%81%A3%E6%B5%B7%E6%B4%8B%E5%A4%A7%E5%AD%B8%22&extralimitunit=%E5%9C%8B%E7%AB%8B%E8%87%BA%E7%81%A3%E6%B5%B7%E6%B4%8B%E5%A4%A7%E5%AD%B8)
* [ISO 9001 dan Good Manufacturing Practices (GMP)](https://trustmandiri.com/perbedaan-iso-9001-dan-gmp-standar-penting-dalam-industri-manufaktur/)
* [Dataset Roboflow ](https://universe.roboflow.com/personcountingsonu/sonu_person-20may)
