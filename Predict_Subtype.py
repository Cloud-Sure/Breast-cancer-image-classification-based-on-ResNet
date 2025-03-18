import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# TODO: Load the trained model
model = tf.keras.models.load_model(
    r'D:\Your\model_path\classifier.h5')
# model = tf.keras.models.load_model(r'D:\down\pythonProject\400X__benign_ResNet152_.h5')

# Load the category label file
class_labels_file = r'D:\down\pythonProject\benign.json' 
with open(class_labels_file, 'r') as f:
    class_labels = json.load(f)

# Assume multiple image paths (batch prediction)
# TODO: Set image_dir
image_dir = r'D:\BIA\test2'  # directory path
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

# ResNet Enter size
image_size = (224, 224)

# Mean and standard deviation during training
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# Batch load and preprocess images
def preprocess_images(image_paths):
    img_arrays = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        img_arrays.append(img_array)
    img_arrays = np.array(img_arrays)

    # Standardization: According to mean and std during training
    img_arrays = img_arrays / 255.0  # normalization
    img_arrays = (img_arrays - mean) / std  # standardization

    return img_arrays


def batch_predict(image_paths):
    # Preprocessed image
    img_arrays = preprocess_images(image_paths)

    # Class prediction
    predictions = model.predict(img_arrays)
    predicted_classes = np.argmax(predictions, axis=1)  # Gets the index value of the prediction

    # Converts index values to labels
    predicted_labels = [class_labels.get(str(predicted_class), "Unknown") for predicted_class in predicted_classes]
    return predicted_labels


# Batch prediction
predicted_labels = batch_predict(image_paths)

# Output prediction result
for img_path, label in zip(image_paths, predicted_labels):
    print(f"Image: {img_path} Prediction Result: {label}")