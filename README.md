# Facial Emotion Recognition with FER2013 and VGG16

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface\&logoColor=white)](https://huggingface.co/shivampr1001/Emo0.1)

This project is a Convolutional Neural Network (CNN) model that detects emotions from facial images. Using the FER2013 dataset, the model classifies facial expressions into seven classes: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**. The model is built with TensorFlow and leverages transfer learning with the pre-trained VGG16 architecture.

---

## 📸 Sample Output

The model detects emotions on various faces, as shown in the image below:

![Sample Output](https://github.com/shivamprasad1001/humanEmotionsRecognition/blob/main/predictedImg.jpg)

---

## 🔥 Easy Usage with Hugging Face Model

You can now **use the pre-trained model directly** without training it yourself. Just clone this repository and run the scripts below — the model will be automatically downloaded from [Hugging Face](https://huggingface.co/shivampr1001/Emo0.1).

### 🔍 Emotion Recognition by Image

```bash
git clone https://github.com/shivamprasad1001/humanEmotionsRecognition.git
cd humanEmotionsRecognition
python predictBY_img.py
```

### 🎥 Real-Time Emotion Recognition via Webcam

```bash
python RealTimeClassification.py
```

> ✅ These scripts will automatically load the pre-trained model from Hugging Face and perform prediction on input images or webcam feed.

> 🔗 **Model hosted at:** [shivamprasad1001/Emo0.1](https://huggingface.co/shivampr1001/Emo0.1)

---

## 📁 Project Structure

* **`modelTrain.py`**: Script for training and evaluating the model on the FER2013 dataset. Running this file generates a pre-trained model file, `Emo0.1.h5`.
* **`predictBY_img.py`**: Python script to predict emotion from an image using the pre-trained Hugging Face model.
* **`RealTimeClassification.py`**: Python script for real-time webcam emotion recognition using the Hugging Face model.
* **`facial_EmotionModel.h5`**: Trained model saved after running `modelTrain.py`.

---

## 📊 Dataset

* **FER2013**: Contains 48x48 grayscale images in `.jpg` format, split into training and testing sets.
* **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🧠 Model Architecture

1. **Base Model**: VGG16 pre-trained on ImageNet, with the convolutional layers frozen initially.
2. **Custom Layers**:

   * Flatten layer for feature extraction
   * Dense layers with dropout for classification

---

## ⚙️ Setup and Installation

### Requirements

* Python 3.x
* TensorFlow
* Numpy
* Matplotlib
* OpenCV
* Transformers (for Hugging Face support)

Install dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python transformers
```

---

## 📌 Usage (Training from Scratch)

1. **Train the Model**:

   ```bash
   python modelTrain.py
   ```

2. **Real-Time Emotion Detection**:
   Open `realTimeRecognition.py` in python script.

3. **Image-Based Emotion Recognition**:
   Open `ByImg.py` in python script.

---

## 🏋️ Training

The model is trained with data augmentation to improve generalization. Early stopping and checkpointing are included.

### Training Parameters

* **Epochs**: 30
* **Batch Size**: 32
* **Learning Rate**: 0.0001

### Sample Model Summary

```
Total params: 14,781,255 (56.39 MB)
Trainable params: 66,567 (260.03 KB)
Non-trainable params: 14,714,688 (56.13 MB)
```

---

## 📈 Monitor with TensorBoard

Run the following to view training logs:

```bash
tensorboard --logdir=logs/fit
```

---

## 🛠️ Troubleshooting

* **Class Imbalance**: Tune `class_weight` to balance under-represented classes.
* **Low Accuracy?** Try ResNet or EfficientNet architectures.

---

## 🤝 Contributing

Contributions are welcome! Submit a pull request or open an issue for improvements.

---

