import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc
import albumentations as A
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Enable DirectML (optional, but recommended to ensure optimizations)
os.environ['TF_ENABLE_DIRECTML_OPTS'] = '1'

# Define image size and batch size
image_size = (224, 224)  # ResNet input size
batch_size = 32

# TODO: Choose class_labels

# benign
class_labels = {
    0: "adenosis",
    1: "fibroadenoma",
    2: "phyllodes_tumor",
    3: "tubular_adenoma"
}

# malignant
class_labels = {
    0: "ductal_carcinoma",
    1: "lobular_carcinoma",
    2: "mucinous_carcinoma",
    3: "papillary_carcinoma"
}



# Function to load and preprocess images with albumentations transformations
def load_and_augment_image(image_path, transform=None):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    if transform:
        augmented = transform(image=img_array)
        return augmented['image']  # Return augmented image as numpy array
    return img_array


# Calculate class weights for imbalanced datasets
def calculate_class_weights(y_train):
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_labels.values()))
    y_train_encoded = label_encoder.transform(y_train)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )
    return dict(enumerate(class_weights))


# Image generator function
def image_generator(image_paths, labels, batch_size, transform=None):
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_labels.values()))

    while True:
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            index = np.random.choice(len(image_paths))
            image_path = image_paths[index]
            label = labels[index]
            image = load_and_augment_image(image_path, transform)
            batch_images.append(image)
            batch_labels.append(label)

        batch_images = np.array(batch_images)
        batch_labels = label_encoder.transform(batch_labels)  # Convert to integer labels
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=len(class_labels))

        yield batch_images, batch_labels


# Function to train a model for a specific dataset path
def train_and_evaluate_model(train_dir, dataset_name):
    # Get image paths and labels from the dataset
    image_paths = []
    image_labels = []
    for class_name in os.listdir(train_dir):
        class_folder = os.path.join(train_dir, class_name)
        if os.path.isdir(class_folder):
            for image_file in os.listdir(class_folder):
                image_paths.append(os.path.join(class_folder, image_file))
                image_labels.append(class_name)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(image_paths, image_labels, test_size=0.3, stratify=image_labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333,
                                                    stratify=y_temp)  # 20% validation, 10% test

    # Albumentations for data augmentation
    train_transforms = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.Transpose(),
        A.Rotate(limit=15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Calculate class weights for the training set
    class_weight_dict = calculate_class_weights(y_train)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_labels.values()))
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert labels to one-hot encoding
    y_train_encoded_one_hot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(class_labels))
    y_val_encoded_one_hot = tf.keras.utils.to_categorical(y_val_encoded, num_classes=len(class_labels))
    y_test_encoded_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(class_labels))

    # TODO: Choose Model
    # Load ResNet model as base model
    base_model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Defrost the last few layers of ResNet
    for layer in base_model.layers[-10:]:  # Defrost the last 10 layers
        layer.trainable = True

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(len(class_labels), activation='softmax')
    ])

    # Compile the model
    # TODO: set learning_rate
    optimizer = Adam(learning_rate=1e-5)
    loss_fn = CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Learning rate scheduler
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.95  # Reduce learning rate after 10 epochs

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    history = model.fit(
        image_generator(X_train, y_train, batch_size, transform=train_transforms),
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_data=image_generator(X_val, y_val, batch_size, transform=val_transforms),
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler, early_stopping]
    )

    # Save model
    # TODO: Rename Model
    model.save(f'{dataset_name}_malignant_ResNet152_.h5')

    # Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.plot(history.history['loss'], label='Training Loss', linestyle='--', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='orange')
    plt.title(f'Training and Validation Accuracy and Loss - {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # TODO: Rename Figure
    plt.savefig(f'{dataset_name}_malignant_ResNet152_ACC&Loss.png')
    plt.close()

    # Predict and calculate ROC curve
    y_pred = []
    y_true = []

    for batch_images, batch_labels in image_generator(X_test, y_test, batch_size, transform=val_transforms):
        batch_predictions = model.predict(batch_images)
        y_pred.extend(batch_predictions)
        y_true.extend(batch_labels)

        if len(y_true) >= len(y_test_encoded_one_hot):
            break

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(12, 8))
    for i in range(len(class_labels)):
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance', linestyle='--', color='gray')
    plt.title(f'ROC Curve for All Classes - {dataset_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    # TODO: Rename Figure
    plt.savefig(f'{dataset_name}_malignant_ResNet152_Roc.png')
    plt.close()


# Batch process for each dataset (40X, 100X, 200X, 400X)
# TODO: set train_dir
base_dir = r'D:\down\pythonProject\output-new\malignant'


for dataset in ['40X', '100X', '200X', '400X']:
    dataset_path = os.path.join(base_dir, dataset)
    train_and_evaluate_model(dataset_path, dataset)
