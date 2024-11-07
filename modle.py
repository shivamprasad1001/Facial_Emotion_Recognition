# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Paths to the FER2013 dataset
train_dir = 'data/fer2013/train'
test_dir = 'data/fer2013//test'

# Set up data augmentation and normalization
datagen =ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='rgb',  # Converts grayscale to RGB by replicating channels
    batch_size=32,
    class_mode='sparse'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='rgb',
    batch_size=32,
    class_mode='sparse'
)

# Load VGG16 as the base model.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the final model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes for emotion recognition
])

# Compile the model only once
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Pass class weights to model.fit
model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    class_weight=class_weights_dict,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


model.save("facial_EmotionModel.h5")
print("model save successfully..")
