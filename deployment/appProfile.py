import streamlit as st
from PIL import Image

def app():
    # Page Title
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>😷 Selamat Datang di <span style='color:#FF5733'>PPEye</span></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Sistem AI untuk Deteksi Kepatuhan Alat Pelindung Diri (APD)</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # Latar Belakang Section
    st.markdown("### 📌 Latar Belakang")
    st.markdown("""
    PT Mayora Indah Tbk menghadapi tantangan serius dalam menjaga kepatuhan penggunaan APD di area produksi — sesuai standar **GMP** dan **ISO 9001**.  
    Pengawasan manual seringkali **tidak efektif**, memakan waktu, dan berisiko tinggi terhadap kualitas serta keamanan produk.

    🧠 Oleh karena itu, kami hadirkan solusi cerdas berupa sistem AI bernama **PPEye**, untuk membantu perusahaan:
    - ✅ Mendeteksi kelengkapan APD secara otomatis
    - ✅ Meningkatkan kepatuhan pekerja
    - ✅ Menjaga standar keamanan pangan & produk
    """)

    st.markdown("---")

    # Apa itu PPEye?
    st.markdown("### 🤖 Apa itu PPEye?")
    st.markdown("""
    **PPEye** adalah sistem berbasis **Artificial Intelligence** yang menggunakan **object detection (YOLOv8)**  
    untuk mendeteksi keberadaan dan kelengkapan APD karyawan di area produksi secara **real-time**.

    🎯 Tujuan utamanya adalah menciptakan lingkungan kerja yang:
    - Aman
    - Sesuai standar industri
    - Bebas kesalahan pengawasan manual
    """)

    st.markdown("---")

    from PIL import Image

    # Tim Kami
    st.markdown("### 👨‍💻 Tim Kami")
    st.markdown("<div style='text-align: center; color: gray;'>Orang-orang hebat di balik pengembangan PPEye</div>", unsafe_allow_html=True)
    st.markdown("")

    # Baris 1
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open("images/pesoy.jpg"), width=200, caption="Gregorius Yoseph Radityo – Project Manager, Data Engineer \n\n Memimpin tim dan bertanggung jawab atas keseluruhan pengembangan proyek")

    with col2:
        st.image(Image.open("images/bilal.jpg"), width=200, caption="Putra Fisabil Muhammad – Data Engineer, Data Scientist, Data Anotator  \n\n Mengembangkan model deteksi YOLO dan mengelola dataset pelatihan.")
        st.markdown("<div style='text-align: center;'></div>", unsafe_allow_html=True)

    # Baris 2
    col3, col4 = st.columns(2)
    with col3:
        st.image(Image.open("images/rafi.jpg"), width=200, caption="Rafi Arya Siregar – Data Analyst, Data Anotator \n\n Membantu analisis data dan memberi label dataset pelatihan.")

    with col4:
        st.image(Image.open("images/rifqi.png"), width=200, caption="Rifqi Munif Imanullah – Data Scientist, Data Anotator \n\n Mengawasi proses testing sistem dan menjamin kualitas aplikasi.")


    st.markdown("---")

    # Kontak
    st.markdown("### 📞 Kontak Kami")
    st.markdown("""
    Jika Anda ingin berdiskusi lebih lanjut atau tertarik dengan implementasi sistem PPEye, silakan hubungi kami:

    - 📧 **Email:** [bilalputra17@gmail.com](mailto:bilalputra17@gmail.com)  
    """)

    st.markdown("---")
    st.success("Terima kasih telah mengunjungi PPEye! 👏")