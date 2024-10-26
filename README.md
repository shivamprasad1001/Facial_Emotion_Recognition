

# Human Emotions Recognition

This project focuses on recognizing human emotions from facial images using deep learning. The model classifies images into seven emotional categories (e.g., happy, sad, anger, surprise), which can be applied to sentiment analysis, mental health monitoring, customer service, and more.

## Features
- **Emotion Classification:** Detects and categorizes emotions from images.
- **Real-time Prediction:** Supports real-time emotion recognition using a webcam.
- **User-friendly Interface:** Simple design for easy access and testing.
- **Dataset Augmentation:** Uses data augmentation to improve model robustness.

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow (using CNN and Sequential APIs)
- NumPy, Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shivamprasad1001/humanEmotionsRecognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd humanEmotionsRecognition
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/). You can download it from there and place it in the `data` directory. This dataset contains labeled facial expressions for training and testing.
## Emotion Categories

The following emotions are recognized by the system, along with their corresponding numeric labels:

| Emotion     | Label |
|-------------|-------|
| Angry       | 0     |
| Disgusted   | 1     |
| Fearful     | 2     |
| Happy       | 3     |
| Neutral     | 4     |
| Sad         | 5     |
| Surprised   | 6     |

These labels are used internally by the model to classify the detected emotions from input images or video streams.


### Model Overview
- This model is built using **Convolutional Neural Networks (CNN)** and the **Sequential API from TensorFlow**.
- It is trained on **seven emotion categories** Angry , Disgusted, Fearful, Happy, Neutral, Sad, Surprised .
- The current model may not achieve high accuracy immediately; however, further training and optimization on the dataset can yield improved results. Contributors are encouraged to experiment with different techniques to enhance performance.

### Usage
1. Run the main script to start the emotion recognition model:
   ```bash
   python main.py
   ```
2. To use the webcam for real-time emotion detection, ensure your system has an active camera, then execute:
   ```bash
   python real_time_recognition.py
   ```

## Contributing
We welcome contributions to improve this model’s accuracy and effectiveness! Here’s how to get started:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Added feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

### Note
This is a great opportunity to enhance your skills in machine learning and computer vision. Feel free to train and optimize the model on the dataset to achieve better accuracy!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or collaboration, feel free to reach out or raise an issue in the repository.

---
