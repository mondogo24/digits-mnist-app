import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model("digits_mnist.keras")

# Crear la interfaz de usuario
st.title("Clasificador dígitos MNIST")
st.write("Sube una imagen de un digito MNIST para clasificarla.")

uploaded_file = st.file_uploader("Sube una imagen de un dígito", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array -= 0.5
    img_array = img_array.reshape(1, 784)

    # Mostrar la imagen subida
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Predicción
    prediction = model.predict(img_array)
    classes = ["0", "1", "2", "3", "4",
               "5", "6", "7", "8", "9"]
    st.write("Predicción:", classes[np.argmax(prediction)])
