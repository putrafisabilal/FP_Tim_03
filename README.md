# PPEye Vision: SOP Compliance Monitor

## Repository Outline

Berikut adalah gambaran singkat dari file-file yang ada di repository saya.

```
1. description.md - Dokumentasi spesifik Milestone
2.
```

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

Di proyek ini, saya mengembangkan pipeline untuk membangun model deteksi objek yang bertujuan memeriksa kepatuhan pekerja terhadap Standar Operasional Prosedur (SOP) di pabrik, khususnya pada penggunaan peralatan pelindung diri (PPE) seperti hairnet, apron, mask, dan gloves. Alat ini dirancang untuk ditempatkan di pintu masuk ruang produksi, memastikan pekerja telah mengenakan pakaian sesuai SOP sebelum memulai aktivitas. Proses dimulai dengan pengumpulan dataset yang beragam: kami memanfaatkan dataset awal dari situs web, dilengkapi dengan data tambahan dari berbagai sumber, termasuk gambar nyata serta foto asli yang kami ambil sendiri, untuk menciptakan dataset yang representatif dan relevan.

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
* [Dataset Roboflow ](https://universe.roboflow.com/personcountingsonu/sonu_person-20may)
