import cv2
import numpy as np
import os
# from tensorflow.keras.models import load_model

# # Load model at module level (only once)
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'unet_model.h5')
# try:
#     # unet_model = load_model(MODEL_PATH)
#     unet_model = load_model( MODEL_PATH, compile=False, safe_mode=False)
#     print(f"Successfully loaded model from {MODEL_PATH}")
# except Exception as e:
#     raise RuntimeError(f"Failed to load U-Net model: {str(e)}")

from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'unet_model.keras')

try:
    unet_model = load_model(MODEL_PATH, compile=False)
    print("U-Net model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Segmentation model load failed: {e}")
def smart_resize(image, target_size=(128, 128)):
    """Resizes with aspect ratio preservation using zero-padding"""
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    delta_h = target_size[0] - new_h
    delta_w = target_size[1] - new_w
    
    padded = cv2.copyMakeBorder(resized,
                              delta_h//2, delta_h - delta_h//2,
                              delta_w//2, delta_w - delta_w//2,
                              cv2.BORDER_CONSTANT, value=0)
    return padded

def segment_image(image_path):
    """Processes any X-ray image to segmentation mask"""
    try:
        # Read and validate
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image file")
        
        # Preprocess
        processed = smart_resize(image)
        processed = processed.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(processed, axis=(0, -1))
        
        # Predict
        mask = unet_model.predict(input_tensor)[0]
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Resize back to original for clinical use
        final_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        return final_mask
        
    except Exception as e:
        raise ValueError(f"Segmentation failed: {str(e)}")






# import cv2
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# # =========================
# # PATH
# # =========================
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'model', 'unet_model.keras')

# # =========================
# # LOAD MODEL
# # =========================
# try:
#     unet_model = load_model(MODEL_PATH, compile=False)
#     print("✅ U-Net model loaded successfully")
# except Exception as e:
#     raise RuntimeError(f"❌ Segmentation model load failed: {e}")

# # =========================
# # MODEL INPUT CONFIG
# # =========================
# INPUT_SHAPE = unet_model.input_shape  # (None, 128, 128, 1)
# TARGET_SIZE = INPUT_SHAPE[1:3]

# print(f"📌 U-Net expects input size: {TARGET_SIZE}")

# # =========================
# # SMART RESIZE (KEEP ASPECT RATIO)
# # =========================
# def smart_resize(image, target_size):
#     h, w = image.shape[:2]

#     scale = min(target_size[0] / h, target_size[1] / w)
#     new_h, new_w = int(h * scale), int(w * scale)

#     resized = cv2.resize(image, (new_w, new_h))

#     delta_h = target_size[0] - new_h
#     delta_w = target_size[1] - new_w

#     padded = cv2.copyMakeBorder(
#         resized,
#         delta_h // 2, delta_h - delta_h // 2,
#         delta_w // 2, delta_w - delta_w // 2,
#         cv2.BORDER_CONSTANT,
#         value=0
#     )

#     return padded

# # =========================
# # SEGMENT FUNCTION
# # =========================
# def segment_lung(image_path):
#     try:
#         # Read image
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise ValueError("Invalid image file")

#         original_h, original_w = image.shape

#         # Preprocess
#         processed = smart_resize(image, TARGET_SIZE)
#         processed = processed.astype(np.float32) / 255.0

#         # Add batch + channel dims
#         input_tensor = np.expand_dims(processed, axis=(0, -1))

#         # Predict
#         mask = unet_model.predict(input_tensor, verbose=0)[0]

#         # Threshold
#         binary_mask = (mask > 0.5).astype(np.uint8) * 255

#         # Resize back to original size
#         final_mask = cv2.resize(binary_mask, (original_w, original_h))

#         return final_mask

#     except Exception as e:
#         raise ValueError(f"Segmentation failed: {str(e)}")
