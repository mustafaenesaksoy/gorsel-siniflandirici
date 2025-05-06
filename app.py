import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Başlık
st.title("📸 Yapay Zeka Destekli Görüntü Sınıflandırıcı")

# Modeli yükle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/my_model.h5")
    return model

model = load_model()

# Sınıf isimlerini al
class_names = sorted(os.listdir("data/fruits/Training")) + sorted(os.listdir("data/animals-10"))
class_names = list(set(class_names))  # Aynı isim varsa birleştir
class_names.sort()

# Görsel yükleme
uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Görseli işle
    img = image.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 100, 100, 3)

    # Tahmin butonu
    if st.button("📊 Tahmin Et"):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"Tahmin Edilen Sınıf: **{predicted_class}**")
        st.info(f"Model Güveni: %{confidence * 100:.2f}")
