# Facial Emotion Recognition with FER2013 and VGG16

This project is a Convolutional Neural Network (CNN) model that detects emotions from facial images. Using the FER2013 dataset, the model classifies facial expressions into seven classes: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**. The model is built with TensorFlow and leverages transfer learning with the pre-trained VGG16 architecture.
## Sample Output

The model detects emotions on various faces, as shown in the image below:

![Sample Output](https://github.com/shivamprasad1001/humanEmotionsRecognition/blob/main/predictedImg.jpg)

## Project Structure

- **`modelTrain.py`**: Script for training and evaluating the model on the FER2013 dataset. Running this file generates a pre-trained model file, `facial_EmotionModel.h5`.
- **`realTimeRecognition.ipynb`**: Notebook for real-time emotion detection using a webcam or video stream.
- **`ByImg.ipynb`**: Notebook for recognizing emotions from individual images.
- **`facial_EmotionModel.h5`**:  This file is generated after running **`modelTrain.py`**. Once generated, it can be used for emotion recognition in other scripts without needing to retrain the model.

## Dataset

- **FER2013**: Contains 48x48 grayscale images in `.jpg` format, split into training and testing sets.
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Model Architecture

1. **Base Model**: VGG16 pre-trained on ImageNet, with the convolutional layers frozen initially.
2. **Custom Layers**:
   - Flatten layer for feature extraction
   - Dense layers with dropout for classification

## Setup and Installation

### Requirements

- Python 3.x
- TensorFlow
- Numpy
- Matplotlib
- OpenCV (for real-time detection)

Install dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

### Usage

1. **Train the Model**:
   Run `model.py` to train the model on the FER2013 dataset. This will generate the `facial_EmotionModel.h5` file:
   ```bash
   python model.py
   ```

2. **Real-Time Emotion Detection**:
   Open `realTimeRecognition.ipynb` and run it in Jupyter Notebook to detect emotions in real time using the pre-trained model.

3. **Image-Based Emotion Recognition**:
   Open `ByImg.ipynb` to recognize emotions from a single image using `facial_EmotionModel.h5`.

## Training

The model is trained with data augmentation to enhance its performance on the FER2013 dataset. Early stopping and checkpointing are implemented to save the best-performing model.

### Sample Training Parameters

- **Epochs**: 30
- **Batch Size**: 32
- **Learning Rate**: 0.0001

## Results

Final model performance (accuracy and loss) is displayed after training. The model can be evaluated on test data within `model.py`.

### Sample Model Summary

```
Total params: 14,781,255 (56.39 MB)
Trainable params: 66,567 (260.03 KB)
Non-trainable params: 14,714,688 (56.13 MB)
```


## Monitoring Training

This project includes TensorBoard support for monitoring metrics like loss and accuracy during training.

To view TensorBoard:
```bash
tensorboard --logdir=logs/fit
```

## Troubleshooting

- **Class Imbalance**: Adjust `class_weight` parameter to account for any imbalance in the dataset.
- **Improving Accuracy**: Experiment with different architectures like ResNet or EfficientNet if accuracy is insufficient.

## Contributing

Contributions are welcome! Please submit a pull request for any improvements.

---
