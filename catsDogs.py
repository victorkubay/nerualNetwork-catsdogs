from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))

if num_classes == 2:
    class_mode = "binary"
else:
    class_mode = "categorical"

# Add data augmentation to help with small dataset
train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    # batch_size=32,
    batch_size=8,
    class_mode=class_mode,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    # batch_size=32,
    batch_size=8,
    class_mode=class_mode,
    subset="validation"
)

class_weights = train_data.class_indices

# # CREATING MODEL
# model = Sequential([
#     Input(shape=(128,128,3)),
#     Conv2D(32, (3,3), activation = "relu", kernel_regularizer=l2(0.01)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation="relu"),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation="relu"),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(512, activation="relu"),
#     Dense(256, activation="relu"),
#     Dense(1, activation="sigmoid") if class_mode == "binary"
#         else Dense(num_classes, activation="softmax")
# ])

model = Sequential([
    Input(shape=(128,128,3)),
    Conv2D(32, (3,3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(), Activation("relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),  # Added dropout
    Conv2D(64, (3,3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(), Activation("relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),  # Added dropout
    Conv2D(64, (3,3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(), Activation("relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),  # Added dropout
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Added dropout
    Dense(1, activation="sigmoid") if class_mode == "binary"
        else Dense(num_classes, activation="softmax")])

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # Freeze for feature extraction

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x) if class_mode == "binary" else Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

loss_function = "binary_crossentropy"
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# #compiling model
# if class_mode == "binary":
#     loss_function = "binary_crossentropy"
# else: class_mode = "categorical_crossentropy"
# model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

# training
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# assessing accuracy of model
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Model accuracy on validation data: {test_accuracy:.2f}")

#saving
model.save("image_classifier.h5")
#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
# from tensorflow.keras.regularizers import l2
# import os
#
# DATASET_PATH = "dataset/"
# num_classes = len(os.listdir(DATASET_PATH))
# class_mode = "binary" if num_classes == 2 else "categorical"
#
# # Add data augmentation to help with small dataset
# train_datagen = ImageDataGenerator(
#     rescale=1/255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )
#
# train_data = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(128,128),
#     batch_size=8,  # Reduced batch size for small dataset
#     class_mode=class_mode,
#     subset="training"
# )
#
# val_data = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(128,128),
#     batch_size=8,
#     class_mode=class_mode,
#     subset="validation"
# )
#
# # Improved model with regularization
# model = Sequential([
#     Input(shape=(128,128,3)),
#     Conv2D(32, (3,3), activation="relu", kernel_regularizer=l2(0.01)),
#     MaxPooling2D(2,2),
#     Dropout(0.25),  # Added dropout
#     Conv2D(64, (3,3), activation="relu", kernel_regularizer=l2(0.01)),
#     MaxPooling2D(2,2),
#     Dropout(0.25),  # Added dropout
#     Flatten(),
#     Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
#     Dropout(0.5),  # Added dropout
#     Dense(1, activation="sigmoid") if class_mode == "binary"
#         else Dense(num_classes, activation="softmax")
# ])
#
# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy" if class_mode == "binary" else "categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# # Train with more epochs and early stopping
# from tensorflow.keras.callbacks import EarlyStopping
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=50,  # Increased epochs
#     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
# )
#
# # Evaluation
# test_loss, test_accuracy = model.evaluate(val_data)
# print(f"Model accuracy on validation data: {test_accuracy:.2f}")
#
# # Save model
# model.save("image_classifier.h5")
