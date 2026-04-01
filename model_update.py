# from tensorflow.keras.models import load_model

# # Convert segmentation model
# seg_model = load_model("app/model/unet_model.keras", compile=False)
# seg_model.save("segmentation_fixed.h5")

# # Convert classification model
# cls_model = load_model("app/model/classifier_model.keras", compile=False)
# cls_model.save("classification_fixed.h5")

# print("✅ Both models converted successfully")


# from tensorflow.keras.models import load_model
# from tensorflow.keras import Input, Model

# # load model (ignore compile issues)
# model = load_model("app/model/unet_model.h5", compile=False)

# # rebuild model cleanly
# model.save("unet_fixed.h5", save_format="h5")

# print("✅ Fixed model saved")


from tensorflow.keras.models import load_model

# ==============================
# 🔹 Convert U-Net (Segmentation)
# ==============================
print("Loading UNet model...")
unet_model = load_model("app/model/unet_fixed.h5", compile=False)

print("Saving compatible UNet model...")
unet_model.save("app/model/unet_compatible.h5", save_format="h5")


# ==============================
# 🔹 Convert Classifier Model
# ==============================
print("Loading Classifier model...")
classifier_model = load_model("app/model/classifier_model.h5", compile=False)

print("Saving compatible Classifier model...")
classifier_model.save("app/model/classifier_compatible.h5", save_format="h5")


print("✅ Conversion completed successfully!")