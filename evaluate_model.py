

import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === 1. Load the trained model ===
model = load_model("app/model/unet_model.h5", compile=False)

# === 2. Define Metrics ===
def dice_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# === 3. List of Diseases ===
diseases = ['COVID', 'Viral Pneumonia', 'Lung_Opacity', 'Normal']
dataset_dir = "dataset/COVID-19_Radiography_Dataset"
image_size = (128, 128)

# === 4. Loop over each disease category ===
for disease in diseases:
    print(f"\n==============================")
    print(f"Evaluating for: {disease}")
    print(f"==============================")

    image_dir = os.path.join(dataset_dir, disease, "images")
    mask_dir = os.path.join(dataset_dir, disease, "masks")

    X_test, Y_test = [], []

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.png', '.jpg')):
            continue

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, image_size) / 255.0
        mask = cv2.resize(mask, image_size) / 255.0

        X_test.append(np.expand_dims(img, axis=-1))
        Y_test.append(np.expand_dims(mask, axis=-1))

    if len(X_test) == 0:
        print(f"No data found for {disease}. Skipping...")
        continue

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # === Predict and threshold ===
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(np.uint8)

    # === METRICS ===
    acc = np.mean(y_pred == Y_test)
    dice = dice_score(Y_test, y_pred)
    iou = iou_score(Y_test, y_pred)

    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Dice Score:     {dice:.4f}")
    print(f"IoU Score:      {iou:.4f}")

    # === CONFUSION MATRIX ===
    print("\nConfusion Matrix:")

     # Convert mask values to 0 or 1
    y_true_f = (Y_test.flatten() > 0.5).astype(np.uint8)
    y_pred_f = (y_pred.flatten() > 0.5).astype(np.uint8)


    cm = confusion_matrix(y_true_f, y_pred_f)
    print(cm)

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        print(f"\nTrue Positives  (TP): {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")
        print(f"True Negatives  (TN): {TN}")
    else:
        print("⚠ Non-binary segmentation mask detected.")

    # === SHOW SAMPLE RESULT ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[0].squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[0].squeeze(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[0].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.suptitle(f"{disease} - Sample Result")
    plt.show()
