

# train_classifier.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = 128
DATASET_PATH = "dataset"

def load_data():
    images = []
    labels = []
    class_names = os.listdir(DATASET_PATH)
    class_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        img_folder = os.path.join(DATASET_PATH, class_name, "images")
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
                images.append(img)
                labels.append(class_map[class_name])

    X = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = to_categorical(np.array(labels))
    return X, y, class_map

def build_classifier(num_classes):
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load & train
X, y, class_map = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_classifier(num_classes=len(class_map))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

model.save("app/model/classifier_model.h5")
print("Classifier model saved to app/model/classifier_model.h5")
print(f" Class Map: {class_map}")
