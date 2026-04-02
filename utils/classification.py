# import cv2
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# # =========================
# # PATHS
# # =========================
# BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'unet_saved_model')

# # MODEL_PATH = os.path.join(BASE_PATH, 'classifier_model.keras')
# # LABELS_PATH = os.path.join(BASE_PATH, 'labels.npy')

# # =========================
# # LOAD MODEL + LABELS
# # =========================
# # try:
# #     classifier_model = load_model(MODEL_PATH, compile=False)
# #     labels = np.load(LABELS_PATH)
# #     print(" Classifier model loaded successfully")
# # except Exception as e:
# #     raise RuntimeError(f" Failed to load classifier model: {e}")

# # =========================
# # MODEL INPUT CONFIG
# # =========================
# INPUT_SHAPE = classifier_model.input_shape  # (None, 128, 128, 3)

# TARGET_SIZE = INPUT_SHAPE[1:3]
# GRAYSCALE = INPUT_SHAPE[-1] == 1

# print(f" Model expects: {TARGET_SIZE}, {'Grayscale' if GRAYSCALE else 'RGB'}")

# # =========================
# # PREPROCESS FUNCTION
# # =========================
# def preprocess_image(image_path):
#     try:
#         # Read image
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if GRAYSCALE else cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Invalid image file")

#         # Resize
#         img = cv2.resize(img, TARGET_SIZE)

#         # Normalize
#         img = img.astype(np.float32) / 255.0

#         # Expand dims
#         if GRAYSCALE:
#             img = np.expand_dims(img, axis=-1)

#         img = np.expand_dims(img, axis=0)

#         return img

#     except Exception as e:
#         raise ValueError(f"Preprocessing failed: {e}")

# # CLASSIFICATION FUNCTION
# def classify_disease(image_path):
#     try:
#         # Preprocess
#         input_img = preprocess_image(image_path)

#         # Predict
#         preds = classifier_model.predict(input_img)
        
#         predicted_idx = int(np.argmax(preds))
#         confidence = float(np.max(preds))

#         predicted_label = labels[predicted_idx]

#         # Severity logic
#         if confidence > 0.9:
#             severity = "Severe"
#         elif confidence > 0.7:
#             severity = "Moderate"
#         else:
#             severity = "Mild"

#         return predicted_label, confidence, severity

#     except Exception as e:
#         raise ValueError(f"Classification failed: {str(e)}")


import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'classifier_saved_model')

try:
    classifier_model = load_model(MODEL_PATH, compile=False)
    print("✅ Classifier model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load classifier model: {e}")

# =========================
# DEFINE LABELS (MANUAL)
# =========================
labels = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']

# =========================
# MODEL INPUT CONFIG (FIXED)
# =========================
# ❌ Removed classifier_model.input_shape
# ✅ Set manually (same as training)
TARGET_SIZE = (128, 128)
GRAYSCALE = False  # change to True if your model was trained on grayscale

print(f"📌 Model expects: {TARGET_SIZE}, {'Grayscale' if GRAYSCALE else 'RGB'}")

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if GRAYSCALE else cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image file")

        img = cv2.resize(img, TARGET_SIZE)
        img = img.astype(np.float32) / 255.0

        if GRAYSCALE:
            img = np.expand_dims(img, axis=-1)

        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

# =========================
# CLASSIFICATION FUNCTION
# =========================
def classify_disease(image_path):
    try:
        input_img = preprocess_image(image_path)

        # ❌ OLD: classifier_model.predict()
        # ✅ NEW: direct call (works for SavedModel)
        preds = classifier_model(input_img, training=False).numpy()

        predicted_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        predicted_label = labels[predicted_idx]

        # Severity logic
        if confidence > 0.9:
            severity = "Severe"
        elif confidence > 0.7:
            severity = "Moderate"
        else:
            severity = "Low"

        return predicted_label, confidence, severity

    except Exception as e:
        raise ValueError(f"Classification failed: {str(e)}")
