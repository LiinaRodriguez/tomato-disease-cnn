import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Cargar el modelo
model = load_model('./models/vgg16_model.h5', compile=False)

# Definir las clases de enfermedades de las hojas de tomate
class_names = [
    "Bacterial Spot", "Early Blight", "Late Blight",  "Leaf Mold",  "Septoria Leaf Mold", 
    "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus",  "Mosaic Virus" , "Healthy"
]

# Título de la aplicación
st.title("Clasificación de Hojas de Tomate")

# Subtítulo
st.write("Cargue una imagen para clasificar la enfermedad de la hoja de tomate.")

# Cargar una imagen
uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocesar la imagen cargada
    img = load_img(uploaded_file, target_size=(64, 64))  # Redimensionar a 64x64
    img_array = img_to_array(img) / 255.0  # Normalizar los valores de píxeles
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión de lote

    col1, col2, col3 = st.columns([1, 2, 1])  # Columna central más grande
    with col2:  # Columna central
        st.image(img, caption="Imagen cargada (64x64)", width=128)  # Mostrar la imagen centrada

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Clase predicha
    confidence = np.max(predictions)  # Probabilidad de la clase predicha

    # Mostrar el resultado de la predicción
    st.write(f"**Predicción:** {class_names[predicted_class]}")
    st.write(f"**Confianza:** {round(100 * confidence, 2)}%")

    # Mostrar el gráfico de barras de las probabilidades
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, predictions[0], color='green')
    plt.title("Probabilidades de Clasificación")
    plt.xlabel("Clases")
    plt.ylabel("Probabilidad")
    plt.xticks(rotation=45, ha='right')  # Rotar nombres de clases
    plt.ylim(0, 1)
    st.pyplot(plt)
