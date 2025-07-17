import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
import tempfile
import time


def app():
    # Title Section
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>🦺 Prediksi Penggunaan APD</h1>
        <h4 style='text-align: center; color: gray;'>Menggunakan Model YOLOv8 untuk Deteksi Otomatis</h4>
        <hr style='border-top: 3px solid #bbb;'>
    """, unsafe_allow_html=True)

    # Model config
    MODEL_PATH = "deployment/best.pt"
    ALLOWED_EXTENSIONS = ['.mov', '.mp4']
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    DEFAULT_CONFIDENCE = 0.5
    RESOLUTION = "1280x720"

    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH, task='detect')

    model = load_model()
    labels = model.names
    bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
                   (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

    # Upload Section
    st.markdown("### 📤 Upload Video Anda (.mp4 / .mov)")
    uploaded_file = st.file_uploader("Upload file", type=['mov', 'mp4'])

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("❌ Ukuran file melebihi 200MB. Silakan unggah file yang lebih kecil.")
            return

        _, ext = os.path.splitext(uploaded_file.name)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            st.error("❌ Format file tidak didukung. Harap unggah file .mov atau .mp4")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.info("⚙️ Sedang memproses video...")

        resW, resH = map(int, RESOLUTION.split('x'))
        cap = cv2.VideoCapture(tmp_file_path)
        output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        recorder = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resW, resH))

        stframe = st.empty()
        fps_display = st.empty()
        object_count_display = st.empty()

        frame_rate_buffer = []
        fps_avg_len = 200
        avg_frame_rate = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.perf_counter()
            frame = cv2.resize(frame, (resW, resH))
            results = model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = detections[i].conf.item()

                if conf > DEFAULT_CONFIDENCE:
                    color = bbox_colors[classidx % 10]
                    label = f'{classname}: {int(conf * 100)}%'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, label_size[1] + 10)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    object_count += 1

            # Display FPS and object count
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Detected Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            recorder.write(frame)

            # Update FPS
            t_stop = time.perf_counter()
            frame_rate_calc = 1 / (t_stop - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = np.mean(frame_rate_buffer)

            fps_display.markdown(f"**🎞️ FPS Rata-rata:** `{avg_frame_rate:.2f}`")
            object_count_display.markdown(f"**📦 Objek Terdeteksi:** `{object_count}`")

        cap.release()
        recorder.release()
        os.unlink(tmp_file_path)

        st.success("✅ Proses selesai! Silakan unduh hasilnya di bawah ini:")
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Unduh Video Hasil",
                data=file,
                file_name="output_detected.mp4",
                mime="video/mp4"
            )
        os.unlink(output_path)

if __name__ == "__main__":
    app()