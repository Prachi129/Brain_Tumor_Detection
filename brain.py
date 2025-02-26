import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pennylane as qml
import h5py
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
IMG_WIDTH, IMG_HEIGHT = 128, 128
NUM_CLASSES = 3  # glioma, meningioma, pituitary tumor
BATCH_SIZE = 32
EPOCHS = 50
N_QUBITS = 8

# Function to load a MATLAB v7.3 .mat file
def load_mat_file_v7_3(filepath):
    with h5py.File(filepath, 'r') as file:
        label = np.array(file['cjdata']['label']).squeeze()
        image = np.array(file['cjdata']['image']).T
        tumor_mask = np.array(file['cjdata']['tumorMask']).T
        return label, image, tumor_mask

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Path to dataset
dataset_path = '/content/dataset/figshare-brain-tumor-dataset/dataset/data'

# Load dataset
images, labels = [], []
for filename in os.listdir(dataset_path):
    if filename.endswith(".mat"):
        filepath = os.path.join(dataset_path, filename)
        label, image, _ = load_mat_file_v7_3(filepath)
        resized_image = resize_image(image, IMG_WIDTH, IMG_HEIGHT)
        images.append(resized_image)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Normalize and preprocess images
images_normalized = images / 255.0
images_expanded = np.expand_dims(images_normalized, axis=-1)
images_rgb = np.repeat(images_expanded, 3, axis=-1)  # Convert to RGB

# One-hot encode labels
encoder = LabelBinarizer()
labels_encoded = encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images_rgb, labels_encoded, test_size=0.2, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Quantum Circuit for Feature Extraction
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_circuit(inputs):
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def quantum_feature_extraction(images):
    quantum_features = []
    for img in images:
        flattened = img.flatten()[:N_QUBITS]
        q_features = quantum_circuit(flattened)
        quantum_features.append(q_features)
    return np.array(quantum_features)

# Extract Quantum Features
X_train_q = quantum_feature_extraction(X_train)
X_test_q = quantum_feature_extraction(X_test)

def build_combined_model():
    # VGG16 Branch
    image_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name="Image_Input")
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers[:-8]:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    vgg_output = Dense(128, activation='relu')(x)

    # Quantum Features Branch
    quantum_input = Input(shape=(N_QUBITS,), name="Quantum_Input")
    q_x = Dense(64, activation='relu')(quantum_input)
    q_output = Dense(32, activation='relu')(q_x)

    # Combined Features
    combined = Concatenate()([vgg_output, q_output])
    final_x = Dense(128, activation='relu')(combined)
    final_x = Dropout(0.5)(final_x)
    output = Dense(NUM_CLASSES, activation='softmax')(final_x)

    # Define and Compile Model
    model = Model(inputs=[image_input, quantum_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and Train Model
combined_model = build_combined_model()
combined_model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

train_inputs = {"Image_Input": X_train, "Quantum_Input": X_train_q}
test_inputs = {"Image_Input": X_test, "Quantum_Input": X_test_q}

history = combined_model.fit(
    train_inputs, y_train,
    validation_data=(test_inputs, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[reduce_lr],
    verbose=1
)

# Evaluate the Model
test_loss, test_accuracy = combined_model.evaluate(test_inputs, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate Reports
y_pred = combined_model.predict(test_inputs)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(classification_report(y_test_labels, y_pred_labels, target_names=['Meningioma', 'Glioma', 'Pituitary Tumor']))

# Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Meningioma', 'Glioma', 'Pituitary Tumor'])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.show()