
# import os
# import numpy as np
# import cv2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split

# # --- U-Net Architecture ---
# def build_unet(input_shape):
#     inputs = Input(input_shape)

#     # Encoder
#     c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)

#     c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
#     c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)

#     # Bottleneck
#     c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
#     c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

#     # Decoder
#     u1 = UpSampling2D((2, 2))(c3)
#     u1 = concatenate([u1, c2])
#     c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
#     c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

#     u2 = UpSampling2D((2, 2))(c4)
#     u2 = concatenate([u2, c1])
#     c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
#     c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

#     model = Model(inputs, outputs)
#     model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # --- Load images and masks from all folders ---
# def load_dataset(dataset_path, img_size=(256, 256)):
#     images = []
#     masks = []
#     for disease_folder in os.listdir(dataset_path):
#         image_path = os.path.join(dataset_path, disease_folder, 'images')
#         mask_path = os.path.join(dataset_path, disease_folder, 'mask_image')
#         for file in os.listdir(image_path):
#             img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
#             mask = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
#             if img is not None and mask is not None:
#                 img = cv2.resize(img, img_size)
#                 mask = cv2.resize(mask, img_size)
#                 images.append(img)
#                 masks.append(mask)
#     return np.array(images), np.array(masks)

# # --- Main ---
# dataset_path = 'dataset/'
# X, Y = load_dataset(dataset_path)
# X = X[..., np.newaxis] / 255.0
# Y = Y[..., np.newaxis] / 255.0

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# model = build_unet((256, 256, 1))
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=8)

# model.save('app/model/unet_model.h5')
# print("U-Net model saved.")

import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- U-Net Architecture ---
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Load dataset ---
def load_dataset(dataset_path, img_size=(256, 256)):
    images, masks = [], []

    covid_base = os.path.join(dataset_path, 'COVID-19_Radiography_Dataset')

    for disease_folder in os.listdir(covid_base):
        disease_path = os.path.join(covid_base, disease_folder)
        if not os.path.isdir(disease_path):
            continue

        image_path = os.path.join(disease_path, 'images')
        mask_path = os.path.join(disease_path, 'masks')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        for file in os.listdir(image_path):
            if file.startswith('.'):
                continue

            img_file = os.path.join(image_path, file)
            mask_file = os.path.join(mask_path, file)

            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                img = cv2.resize(img, img_size)
                mask = cv2.resize(mask, img_size)
                images.append(img)
                masks.append(mask)

    return np.array(images), np.array(masks)

# --- Main ---
dataset_path = '/kaggle/input/lungs-dataset'
X, Y = load_dataset(dataset_path)

X = X[..., np.newaxis] / 255.0
Y = Y[..., np.newaxis] / 255.0

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

model = build_unet((256, 256, 1))
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=8)

model.save('/kaggle/working/unet_model.h5')
print("U-Net model saved as unet_model.h5")
