import streamlit as st
from ultralytics import YOLOv10
from PIL import Image


def main():
    st.title('Helmet Safety Detection Program:')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        TRAINED_MODEL_PATH = 'D:/best.pt'
        model = YOLOv10(TRAINED_MODEL_PATH)
        st.text(model.info())

        results = model.predict(source='anh1.jpg', imgsz=640, conf=0.3)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Uploaded Image")


if __name__ == "__main__":
    main()
