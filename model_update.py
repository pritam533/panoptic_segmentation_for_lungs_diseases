from tensorflow.keras.models import load_model

# Convert segmentation model
seg_model = load_model("app/model/unet_model.keras", compile=False)
seg_model.save("segmentation_fixed.h5")

# Convert classification model
cls_model = load_model("app/model/classifier_model.keras", compile=False)
cls_model.save("classification_fixed.h5")

print("✅ Both models converted successfully")