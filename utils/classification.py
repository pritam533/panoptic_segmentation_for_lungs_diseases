import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model with input shape validation
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'classifier_model.h5')
classifier_model = load_model(CLASSIFIER_PATH)

# Get model's expected input shape
if len(classifier_model.input_shape) == 4:  # CNN (e.g., (None, 120, 120, 1))
    TARGET_SIZE = classifier_model.input_shape[1:3]
    GRAYSCALE = classifier_model.input_shape[-1] == 1
else:  # Flattened input (e.g., (None, 57600))
    TARGET_SIZE = (int(np.sqrt(classifier_model.input_shape[1]/3)), )*2 if not GRAYSCALE else (int(np.sqrt(classifier_model.input_shape[1])),)*2
    GRAYSCALE = classifier_model.input_shape[1] % 3 != 0

print(f"Model expects: {TARGET_SIZE} {'grayscale' if GRAYSCALE else 'RGB'}")

def preprocess_image(image_path):
    """Universal preprocessor adapting to model requirements"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if GRAYSCALE else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Resize and normalize
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0

    # Format for model
    if len(classifier_model.input_shape) == 4:
        return np.expand_dims(img, axis=(0, -1)) if GRAYSCALE else np.expand_dims(img, axis=0)
    else:
        return img.reshape(1, -1)  # Flatten

def classify_disease(image_path):
    try:
        # Preprocess
        input_img = preprocess_image(image_path)
        
        # Predict
        preds = classifier_model.predict(input_img)
        classes = ['Normal', 'COVID-19', 'Pneumonia', 'Tuberculosis']
        
        # Interpret results
        predicted_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        severity = "Severe" if confidence > 0.9 else "Moderate" if confidence > 0.7 else "Mild"
        
        return classes[predicted_idx], confidence, severity
        
    except Exception as e:
        raise ValueError(f"Classification failed: {str(e)}")



