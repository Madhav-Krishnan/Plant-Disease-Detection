# 🧠 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates a complete pipeline for multi-class image classification using **Convolutional Neural Networks (CNN)** with **TensorFlow** and **Keras**. It includes image preprocessing, real-time data augmentation, model training, validation, evaluation, and prediction on new images.

---

## 📌 Overview

- 🔍 Image preprocessing and augmentation using `ImageDataGenerator`
- 🧠 Custom CNN architecture for classification
- 📊 Training with real-time validation
- 🧪 Evaluation with test accuracy and prediction function
- 📈 Modular design suitable for extension with transfer learning (e.g., ResNet, VGG)

---

## 📁 Project Structure

```
image-classification-cnn/
├── dataset/                     # Dataset organized by class folders
├── main.py            # Core training and prediction script
├── requirements.txt             # List of required Python packages
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Your dataset should follow this folder structure:

```
dataset/
├── class_1/
│   ├── img1.jpg
│   └── img2.jpg
├── class_2/
│   ├── img3.jpg
│   └── img4.jpg
...
```

Update the `dataset_path` variable inside `cnn_classifier.py`:

```python
dataset_path = "path_to_your_dataset"
```

### 3. Run the Model

```bash
python cnn_classifier.py
```

The script will:
- Train the CNN model
- Evaluate it on the validation set
- Print the classification accuracy
- Allow predictions on custom images

---

## 🖼️ Predict on New Image

To predict a new image class, replace the path in the script:

```python
img_path = "path_to_new_image.jpg"
print(f"Predicted class: {predict_image(img_path)}")
```

---

## 🧠 Model Architecture

- Input shape: `(256, 256, 3)`
- Layers:
  - `Conv2D` → `MaxPooling2D` × 3
  - `Flatten`
  - `Dense(512)` with ReLU + Dropout
  - Output: `Dense(num_classes)` with Softmax

---

## 📈 Example Output

```
Epoch 25/25
Training accuracy: 0.92
Validation accuracy: 0.89
Predicted class: cat
```

---

## 📋 Requirements

- TensorFlow ≥ 2.8
- NumPy ≥ 1.19

Install with:

```bash
pip install tensorflow numpy
```

---

## 🚀 Future Enhancements

- Integrate pre-trained models (e.g., ResNet50, MobileNet)
- Add support for transfer learning
- Save and load trained models
- Build a front-end using Streamlit or Flask
- Visualize learning curves and confusion matrix

---

## 🧾 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

Developed using:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Python](https://www.python.org/)

---

> ✨ **Build. Train. Evaluate. Predict.**  
> A complete deep learning workflow — all in one place.
