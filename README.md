
# Tomato Leaf Disease Detection

This project aims to detect and classify 10 different types of diseases in tomato leaves using a Convolutional Neural Network (CNN). By analyzing images of tomato leaves, the model will identify the presence of specific diseases or healthy leaf status, aiding in early intervention and treatment.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training and Validation](#training-and-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [Interface](#interface)

## Overview
The project utilizes deep learning techniques for image classification to identify 10 types of tomato leaf conditions:
1. **Tomato Mosaic Virus** (Virus_mosaico_del_tomate)
2. **Target Spot** (Mancha_anillada)
3. **Bacterial Spot** (Mancha_bacteriana)
4. **Tomato Yellow Leaf Curl Virus** (Virus_del_enrollamiento_amarillo_de_la_hoja_del_tomate)
5. **Late Blight** (Tizón_tardío)
6. **Leaf Mold** (Moho_de_hoja)
7. **Early Blight** (Tizón_temprano)
8. **Spider Mites** (Ácaros Ácaros_de_dos_manchas)
9. **Healthy Tomato** (Tomate___saludable)
10. **Septoria Leaf Spot** (Mancha_foliar_por_Septoria)

## Dataset
The dataset includes:
- **Training set**: 10 classes with 1000 images per class.
- **Validation set**: 10 classes with 100 images per class.

Each image is 256x256 pixels.

## Model Architectures
We implement and compare three CNN architectures to identify the most effective one for this task. Each model is evaluated through 3-fold cross-validation to determine average accuracy and standard deviation.

-- ### Name the architectures --

## Training and Validation
The training process involves:
1. **3-Fold Cross-Validation**: For each model, calculate the average accuracy and standard deviation on the training dataset.
2. **Best Model Selection**: The model with the highest average accuracy is trained on the entire training set.
3. **Evaluation on Validation Set**: Performance is measured using accuracy, precision, recall, F1-score, and confusion matrix on the validation set.

## Evaluation Metrics
The model evaluation includes:
- **Accuracy**: The overall correct predictions.
- **Precision**: Correctly predicted positive observations.
- **Recall**: Correctly identified positive observations.
- **F1-Score**: Weighted average of precision and recall.
- **Confusion Matrix**: Breakdown of actual vs. predicted classifications.

## Interface
An interface allows users to upload an image from the validation set to predict the disease or healthy status. The interface is web-based, built in Streamlit.
