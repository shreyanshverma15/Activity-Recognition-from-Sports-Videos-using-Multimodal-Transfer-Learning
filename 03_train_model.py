import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, TimeDistributed, Dense, Flatten, LSTM, Dropout,
    GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# --- PART 1: Scan Local Files ---
NPY_DATA_DIR = "/content/UCF101_Sports_NPY_Full"

class_names = sorted(os.listdir(NPY_DATA_DIR))
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes: {class_names}")

X_spatial_paths = []
X_temporal_paths = []
y_labels = []
label_map = {name: i for i, name in enumerate(class_names)}

print("Scanning for all local .npy files...")
for class_name in class_names:
    class_dir = os.path.join(NPY_DATA_DIR, class_name)
    class_label = label_map[class_name]

    for file_name in os.listdir(class_dir):
        if file_name.endswith("_spatial.npy"):
            base_name = file_name.replace("_spatial.npy", "")
            spatial_path = os.path.join(class_dir, file_name)
            temporal_path = os.path.join(class_dir, f"{base_name}_temporal.npy")

            if os.path.exists(temporal_path):
                X_spatial_paths.append(spatial_path)
                X_temporal_paths.append(temporal_path)
                y_labels.append(class_label)

y_categorical = to_categorical(y_labels, num_classes=NUM_CLASSES)

(X_spatial_train, X_spatial_test,
 X_temporal_train, X_temporal_test,
 y_train, y_test) = train_test_split(
    X_spatial_paths, X_temporal_paths, y_categorical,
    test_size=0.2, random_state=42, stratify=y_categorical
)

print(f"Training samples: {len(X_spatial_train)}")
print(f"Testing samples: {len(X_spatial_test)}")

# --- PART 2: The FAST Data Generator ---
def fast_data_generator(spatial_paths, temporal_paths, labels, batch_size):
    num_samples = len(spatial_paths)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            # Load .npy files from the fast local disk
            batch_X_spatial = [np.load(spatial_paths[i]) for i in batch_indices]
            batch_X_temporal = [np.load(temporal_paths[i]) for i in batch_indices]
            batch_y = labels[batch_indices]

            yield (
                {'spatial_input': np.array(batch_X_spatial),
                 'temporal_input': np.array(batch_X_temporal)},
                np.array(batch_y)
            )

# --- PART 4: Build the Fused Model (This builds it from scratch) ---
SPATIAL_SEQ_LENGTH = 20
TEMPORAL_SEQ_LENGTH = 10
IMG_SIZE = 128

# Spatial stream
spatial_input_shape = (SPATIAL_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
video_input_spatial = Input(shape=spatial_input_shape, name='spatial_input')
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False
cnn_features = TimeDistributed(base_model)(video_input_spatial)
cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
lstm_features_spatial = LSTM(256, dropout=0.5)(cnn_features)

# Temporal stream
temporal_input_shape = (TEMPORAL_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 2)
video_input_temporal = Input(shape=temporal_input_shape, name='temporal_input')
cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(video_input_temporal)
cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
cnn = TimeDistributed(BatchNormalization())(cnn)
cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
cnn = TimeDistributed(BatchNormalization())(cnn)
cnn = TimeDistributed(Flatten())(cnn)
lstm_features_temporal = LSTM(256, dropout=0.5)(cnn)

# Fusion
combined_features = tf.keras.layers.concatenate([lstm_features_spatial, lstm_features_temporal])
x = Dense(1024, activation='relu')(combined_features)
x = Dropout(0.5)(x)
final_output = Dense(NUM_CLASSES, activation='softmax')(x)

fused_model = Model(inputs=[video_input_spatial, video_input_temporal], outputs=final_output)

fused_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
fused_model.summary()

# --- PART 5: Train the Fused Model ---
BATCH_SIZE = 16
EPOCHS = 20 #can increase epochs to get better accuracy but will take more time

train_gen = fast_data_generator(X_spatial_train, X_temporal_train, y_train, BATCH_SIZE)
test_gen = fast_data_generator(X_spatial_test, X_temporal_test, y_test, BATCH_SIZE)

steps_per_epoch = len(X_spatial_train) // BATCH_SIZE
validation_steps = len(X_spatial_test) // BATCH_SIZE

checkpoint_path = "/content/drive/MyDrive/fused_model_checkpoint_FULL_FINAL.keras"

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

print("\nStarting NEW FUSED model training (20 Epochs)...")
history = fused_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_gen,
    validation_steps=validation_steps,
    callbacks=[checkpoint_callback]
)

print(f"Training complete! The best model is saved to {checkpoint_path}")

# --- PART 6: Plot Results ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()
plt.show()
