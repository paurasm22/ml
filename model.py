import os
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


dataset_path = "/root/.cache/kagglehub/datasets/akshitmadan/eyes-open-or-closed/versions/1"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")


# Step 1: Preprocessing Training, Validation, and Test Data
def preprocess_data(train_path, test_path, image_size=(224, 224)):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,  # Augment data
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    print("Class Indices:", train_generator.class_indices)  # Print class mapping

    return train_generator, validation_generator, test_generator

# Step 2: Build ResNet Model
def build_resnet_model(input_shape=(224, 224, 3), fine_tune_at=140):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze initial layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the Model with Callbacks
def train_model(model, train_generator, validation_generator, epochs=20):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)
    model.save('eye_status_resnet_model.h5')

# Step 4: Evaluate the Model
def evaluate_model(model_path, test_generator):
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# train_path = '/kaggle/input/eyes-open-or-closed/dataset/test'
# test_path = '/kaggle/input/eyes-open-or-closed/dataset/test'

train_gen, val_gen, test_gen = preprocess_data(train_path, test_path)

resnet_model = build_resnet_model()
train_model(resnet_model, train_gen, val_gen)

evaluate_model('eye_status_resnet_model.h5', test_gen)