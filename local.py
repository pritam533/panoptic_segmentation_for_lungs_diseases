from tensorflow.keras.models import load_model

model = load_model("app/model/classifier_model.keras", compile=False)
model.save("app/model/classifier_model.h5")