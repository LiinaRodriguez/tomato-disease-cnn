# Tomato Leaf Disease Detection

This project aims to detect and classify 10 different types of diseases in tomato leaves using a Convolutional Neural Network (CNN). By analyzing images of tomato leaves, the model will identify the presence of specific diseases or healthy leaf status, aiding in early intervention and treatment.
## Authors
 - Jhonatan Ferrer
 - Yarlinson Barranco
 - Lina Rodriguez
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training and Validation](#training-and-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
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
8. **Spider Mites** (Ácaros_de_dos_manchas)
9. **Healthy Tomato** (Tomate_saludable)
10. **Septoria Leaf Spot** (Mancha_foliar_por_Septoria)

## Dataset
The dataset includes:
- **Training set**: 10 classes with 1000 images per class.
- **Validation set**: 10 classes with 100 images per class.

Each image was resized to 64x64 pixels for computational efficiency.

## Model Architectures
We implemented and compared three CNN architectures, known for their efficiency in image classification tasks:
1. **EfficientNetB0**
2. **MobileNetV2**
3. **VGG16**

### Configurations
Each model was trained with the following hyperparameter configurations:
1. **Dropout rate**: 0.5 or 0.3
2. **Dense units**: 1024 or 512
3. **Frozen layers**: 50 or 100

The best performing configuration was **VGG16** with:
- Dropout rate: 0.5
- Dense units: 1024
- Frozen layers: 50

## Training and Validation
The training process involved:
1. **3-Fold Cross-Validation**: Each model was trained and evaluated across three partitions to compute average accuracy and standard deviation.
2. **Final Model Training**: The best model configuration (VGG16) was trained on the full training set.
3. **Validation Set Evaluation**: Performance was assessed on the validation set using the metrics below.

## Evaluation Metrics
The following metrics were used to evaluate the models:
- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Proportion of true positives among all predicted positives.
- **Recall**: Proportion of true positives among all actual positives.
- **F1-Score**: Weighted harmonic mean of precision and recall.
- **Confusion Matrix**: Detailed analysis of actual vs. predicted classifications.

## Results
The results of the 3-fold cross-validation are summarized below:

| Model            | Dropout | Dense Units | Frozen Layers | Accuracy  | Std. Dev.   |
|------------------|---------|-------------|---------------|-----------|-------------|
| EfficientNetB0   | 0.5     | 1024        | 50            | 55.70\%   | ± 0.27\%    |
| EfficientNetB0   | 0.3     | 512         | 100           | 16.24\%   | ± 0.03\%    |
| MobileNetV2      | 0.5     | 1024        | 50            | 57.65\%   | ± 0.21\%    |
| MobileNetV2      | 0.3     | 512         | 100           | 60.63\%   | ± 0.36\%    |
| **VGG16**        | 0.5     | 1024        | 50            | **66.87\%** | **± 0.06\%** |
| VGG16            | 0.3     | 512         | 100           | 65.55\%   | ± 0.19\%    |

Final evaluation on the validation set (VGG16 with the best configuration):
- **Accuracy**: 79.10%
- **Precision**: 79.67%
- **Recall**: 79.10%
- **F1-Score**: 78.98%

The confusion matrix highlights that the model performed well on uniformly lit, dark leaves but struggled with textured, brightly colored, or shadowed leaves.

## Interface
An interactive interface, built with Streamlit, allows users to:
1. Upload an image.
2. View the uploaded image for confirmation.
3. Receive a classification result, including probabilities for each disease class.

## Code and Models
- **Training Code**: The code to train all the models is available in the `models/` directory.
- **Interface Code**: Streamlit interface implementation is located in the `src/` directory.

## How to Run

