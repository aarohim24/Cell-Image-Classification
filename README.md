Cell Image Classification Using CNNs

This project demonstrates how to build a Convolutional Neural Network (CNN) to classify microscopic images of blood cells using Python and TensorFlow/Keras. It combines machine learning with biological data to explore how deep learning can assist in medical image analysis.

---

## Dataset

We use the [Blood Cell Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) from Kaggle, which contains over 12,000 labeled images of four types of blood cells:
- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

---

## Setup Instructions (Google Colab)

1. Download the dataset from Kaggle and upload the ZIP file to your Colab session.
2. Extract the dataset using Python's `zipfile` module.
3. Use `ImageDataGenerator` from Keras to load and augment the images.
4. Build and train a CNN model using TensorFlow/Keras.
5. Visualize training results and sample predictions.

---

##  Results

### Accuracy and Loss Plots

*(Insert training and validation accuracy/loss plots here)*

### Sample Predictions

*(Insert sample cell images with predicted and true labels here)*

---

## 🔧 Future Improvements

- Use transfer learning with pre-trained models like ResNet or MobileNet
- Add confusion matrix and classification report
- Explore more complex architectures or hyperparameter tuning
- Deploy the model as a web app using Streamlit or Flask

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib
- Google Colab

---

## Acknowledgements

Dataset by Paul Mooney on Kaggle: [Blood Cell Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

---

Feel free to fork this repo and build on it!
