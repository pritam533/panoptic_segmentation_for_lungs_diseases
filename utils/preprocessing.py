import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess the image for model prediction.
    
    Args:
        image_path (str): Path to input X-ray image.
        target_size (tuple): Size expected by the model.

    Returns:
        preprocessed_img (np.array): Preprocessed image tensor (1, H, W, C)
        original_img (np.array): Original image for display or saving
    """
    # Load image in grayscale (you can also use RGB if model was trained in RGB)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_img = img.copy()

    # Resize to target shape
    img = cv2.resize(img, target_size)

    # Normalize to [0, 1]
    img = img / 255.0

    # Expand dimensions to match model input shape (1, H, W, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img, original_img
