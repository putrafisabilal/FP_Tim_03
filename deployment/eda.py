import streamlit as st
import os
from PIL import Image

# st.set_page_config(page_title="EDA Page", layout="wide")

def app():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("""
    Menganalisis karakteristik data untuk kemudian diimplementasikan pada proses modeling.
    """)

    # folder images
    folder_path = "images"
    
    
    # EDA 1
    st.header("Distribusi Kelas Pada Dataset")

    # Buka gambar dan tampilkan
    eda1 = os.path.join(folder_path, "eda1.png")  # pastikan file ini ada
    image1 = Image.open(eda1)
    st.image(image1, caption="")

    st.markdown("""
                | **Class**   | **Train Data** | **Validation Data** | **Total** |
                | ----------- | -------------- | ------------------- | --------- |
                | No\_Glove  | 1,363          | 338                 | 1,701     |
                | No\_Apron   | 649            | 146                 | 795       |
                | No\_Mask    | 848            | 219                 | 1,067     |
                | No\_Hairnet | 703            | 174                 | 877       |
                | Glove       | 947            | 240                 | 1,187     |
                | Hairnet     | 800            | 205                 | 1,005     |
                | Apron       | 726            | 194                 | 920       |
                | Mask        | 388            | 92                  | 480       |
                | **Total**   | **6,424**      | **1,608**           | **8,032** |

                Berdasarkan hasil distribusi pada dataset, terdapat beberapa insight yang diperoleh:

                1. Jumlah total bounding box pada dataset adalah 8.032 gambar, yang tersebar di berbagai kelas alat pelindung diri (APD).
                2. No_Glove merupakan kelas terbanyak dengan 1.701 data (21.2% dari total dataset), diikuti oleh Glove dengan 1.187 data (14.8%). Selain itu, kelas No_Mask (1.067 data / 13.3%) dan Hairnet (1.005 data / 12.5%) juga memberikan kontribusi signifikan terhadap distribusi data. Di sisi lain, Mask adalah kelas dengan jumlah data terkecil, hanya 480 data (6.0%).
                3. Dataset ini memiliki distribusi data yang proporsional, dengan keseimbangan antar kelas yang cukup baik. Meskipun ada kelas dengan jumlah data yang lebih sedikit (seperti Mask), distribusi kelas secara keseluruhan tetap memberikan gambaran yang cukup baik untuk deteksi objek.

                **Kesimpulan:**

                Dataset yang terkumpul merupakan data yang diambil dari beberapa sumber internet dan hasil gambar yang kami produksi. Secara keseluruhan, dataset yang kami peroleh memiliki proporsi kelas seimbang yang telah mencakup objek-objek penting dalam pendeteksian alat pelindung diri (APD).
                """)
    
    # EDA 2
    st.header("Ukuran Gambar Pada Dataset")

    # Buka gambar dan tampilkan
    eda2 = os.path.join(folder_path, "eda2.png")  # pastikan file ini ada
    image2 = Image.open(eda2)
    st.image(image2, caption="")

    st.markdown("""
                Berikut hasil insight yang dapat diperoleh berdasarkan resolusi gambar pada dataset:

                1. Terdapat variasi ukuran yang signifikan antar kelas. Tinggi gambar bervariasi dari 480 picel hingga 899 pixel. Untuk lebar gambar berkisar antara 596 pixel hingga 1.032 pixel.
                2. Kelas dengan ukuran resolusi gambar terkecil adalah **Hairnet** dengan ukuran sebesar 408 pixel, sementara yang yang paling tinggi adalah kelas **No_Harinet** dengan ukuran tertinggi dengan 899 pixel. Kelas dengan ukuran resolusi terlebar adalah kelas **Mask** dengan ukuran 1.032 pixel, yang bisa disebabkan karena area wajah yang luas.
                3. Terdapat ketidakseimbangan dataset antar kelas seperti No_Mask dan Mask yang berbeda hampir 500 data. 
                """)
    
    # EDA 3
    st.header("Visualisasi Heatmap Setiap Kelas")

    # Buka gambar dan tampilkan
    eda3 = os.path.join(folder_path, "eda3.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.1.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")
    
    eda3 = os.path.join(folder_path, "eda3.2.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.3.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.4.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.5.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.6.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    eda3 = os.path.join(folder_path, "eda3.7.png")  # pastikan file ini ada
    image3 = Image.open(eda3)
    st.image(image3, caption="")

    st.markdown("""
                Berikut hasil anaisis terkait sampling dari 8 kelas beserta heatmap setiap anotasinya:

                **1. Apron**

                - **Distribusi Heatmap**: Titik pusat sebagian besar berada di sekitar bagian tengah tubuh, dengan intensitas lebih tinggi di area dada dan perut.
                - **Observasi**: Memiliki distribusi yang cukup terpusat pada tubuh bagian atas.

                **2. Glove**

                - **Distribusi Heatmap**: Titik pusat lebih terfokus pada tangan, dengan intensitas lebih tinggi di sekitar tangan.
                - **Observasi**: Distribusi lebih tersebar di area tangan dengan variasi sedikit lebih tinggi dibandingkan kelas lain.

                **3. Hairnet**

                - **Distribusi Heatmap**: Titik pusat terfokus di bagian atas kepala, dengan sedikit penyebaran di sekitar area kepala.
                - **Observasi**: Heatmap terpusat pada kepala, khususnya bagian atas.

                **4. Mask**

                - **Distribusi Heatmap**: Titik pusat terfokus pada area wajah, terutama bagian mulut dan hidung.
                - **Observasi**: Distribusi dominan di sekitar wajah, dengan fokus pada area yang menutupi hidung dan mulut.

                **5. No_Apron**

                - **Distribusi Heatmap**: Titik pusat lebih tersebar ke area tubuh bagian atas dengan beberapa variasi pada posisi.
                - **Observasi**: Meskipun tidak ada apron, distribusi heatmap lebih luas dibandingkan dengan kelas "Apron".

                **6. No_Glove**

                - **Distribusi Heatmap**: Titik pusat berada di area tubuh, sering kali sekitar dada atau tangan.
                - **Observasi**: Distribusi lebih tersebar, menunjukkan bahwa tanpa sarung tangan, titik pusat ada di bagian tubuh yang lebih besar.

                **7. No_Hairnet**

                - **Distribusi Heatmap**: Titik pusat terfokus di sekitar kepala, menunjukkan area yang kosong tanpa pelindung rambut.
                - **Observasi**: Heatmap terpusat pada kepala, dengan distribusi yang lebih luas daripada kelas dengan "Hairnet".

                **8. No_Mask**

                - **Distribusi Heatmap**: Titik pusat terfokus pada wajah, terutama area mulut dan hidung tanpa masker.
                - **Observasi**: Distribusi pusat wajah yang tidak tertutup masker terlihat jelas pada area mulut dan hidung.

                **Kesimpulan:**

                - **Kelas dengan fokus terpusat di kepala**: Hairnet, Mask, No_Hairnet, No_Mask.
                - **Kelas dengan distribusi di area tangan**: Glove, No_Glove.
                - **Kelas dengan distribusi tubuh bagian atas**: Apron, No_Apron.
                """)
    
    # EDA 4
    st.header("Heatmap Bounding Box")

    # Buka gambar dan tampilkan
    eda4 = os.path.join(folder_path, "eda4.png")  # pastikan file ini ada
    image4 = Image.open(eda4)
    st.image(image4, caption="")

    st.markdown("""
                Berikut hasil anaisis terkait lokasi paling sering munculnya objek:

                **1. Posisi Paling Sering Munculnya Objek**

                Objek paling sering muncul di sekitar posisi X = 0.5 dan posisi Y = 0.6. Ini menunjukkan bahwa objek cenderung terkumpul di bagian tengah gambar, baik secara horizontal maupun vertikal.

                **2. Kepadatan Objek**

                Warna merah pada gambar menunjukkan kepadatan tinggi objek yang terdeteksi di posisi tertentu. Intensitas tertinggi pada posisi X = 0.5 dan Y = 0.6, yang mengindikasikan posisi utama bagi objek yang dianalisis. Warna putih menandakan kepadatan objek paling tinggi di sekitar posisi tersebut.

                **3. Distribusi Objek**

                Area yang lebih terang pada gambar menunjukkan lebih banyak objek yang terdeteksi. Objek lebih sering muncul di area tengah gambar, dengan konsentrasi lebih tinggi di sekitar posisi 0.5 di sumbu X dan 0.6 di sumbu Y. Terdapat sebaran yang lebih lebar pada posisi horizontal, sementara sebaran vertikal lebih terkonsentrasi.

                **4. Konsistensi Posisi Objek**

                Posisi objek cukup konsisten di bagian tengah gambar, dengan kepadatan objek yang sangat terlihat di sekitar posisi 0.5 X dan 0.6 Y. Area di luar posisi tersebut menunjukkan kepadatan rendah baik di bagian kiri, kanan, atas, maupun bawah gambar.
                """)
    
    # EDA 5
    st.header("Heatmap Distribusi Letak Bounding Box Per Kelas")

    # Buka gambar dan tampilkan
    eda5 = os.path.join(folder_path, "eda5.png")  # pastikan file ini ada
    image5 = Image.open(eda5)
    st.image(image5, caption="")

    st.markdown("""
                Berikut hasil anaisis terkait distribusi titik tengah bounding box:

                **1. Posisi Paling Sering Munculnya Objek**

                - **Apron**: Titik pusat objek paling sering muncul di sekitar posisi X = 0.5 dan Y = 0.6.
                - **Glove**: Titik pusat objek lebih terfokus di sekitar tangan, di posisi X = 0.3 dan Y = 0.6.
                - **Hairnet**: Titik pusat berada di bagian atas kepala, dengan konsentrasi di sekitar X = 0.5 dan Y = 0.2.
                - **Mask**: Titik pusat paling sering di area wajah, dengan konsentrasi utama pada X = 0.6 dan Y = 0.2.

                **2. Kepadatan Objek**

                - Posisi dengan kepadatan objek paling tinggi ditemukan di sekitar X = 0.5 dan Y = 0.6 untuk beberapa kelas seperti **Apron** dan **Glove**.
                - Warna merah pada heatmap menunjukkan area dengan kepadatan objek yang lebih tinggi, terutama pada bagian tengah tubuh dan wajah.

                **3. Distribusi Objek**

                - Pada **No_Apron** dan **No_Glove**, distribusi titik pusat lebih tersebar pada area tubuh bagian atas dan tangan.
                - **Hairnet** dan **Mask** menunjukkan konsentrasi titik pusat di kepala dan wajah dengan lebih sedikit penyebaran horizontal.
                **4. Konsistensi Posisi Objek**

                - Posisi objek pada **Apron**, **Glove**, dan **Mask** menunjukkan konsistensi yang lebih besar di area tengah tubuh dan wajah, masing-masing.
                - **No_Apron** dan **No_Glove** menunjukkan variasi lebih besar, dengan titik pusat objek tersebar lebih luas.

                **5. Variasi Distribusi**

                - **No_Apron** memiliki distribusi titik tengah objek yang lebih lebar. Hal ini menunjukkan ketidakhadiran apron lebih banyak pada tubuh bagian atas.
                - **Hairnet** dan **No_Hairnet** memiliki konsentrasi objek lebih tinggi pada bagian kepala. Hal ini menunjukkan perbedaan dengan dan tanpa pelindung rambut.

                **6. Konsentrasi Titik Tengah**

                - **Mask** memiliki konsentrasi titik tengah objek yang sangat tinggi di area wajah, khususnya pada bagian mulut dan hidung. Hal ini menunjukkan distribusi masker yang terfokus di area tersebut.

                **7. Area Tanpa Objek**

                - **No_Apron** dan **No_Mask** memiliki area besar tanpa objek terutama di bagian kiri atau luar gambar. Hal ini menandakan lebih sedikit objek di bagian tersebut.

                **8. Hubungan Titik Tengah dan Pakaian**

                - **Apron** dan **No_Apron** menunjukkan konsentrasi objek yang lebih banyak di tubuh bagian atas, terutama di sekitar dada dan perut.
                - **Glove** dan **No_Glove** menunjukkan bahwa tangan lebih sering terlihat pada titik pusat dibandingkan dengan bagian tubuh lainnya.
                """)
