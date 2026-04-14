import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ======================
# PARAMETER
# ======================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50  # besar, nanti dihentikan otomatis

DATASET_PATH = "D:\\Semester 6\\Deep Learning\\Projek Deni\\road_surface_dataset"

# ======================
# DATA AUGMENTATION
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ======================
# MODEL (TRANSFER LEARNING)
# ======================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze semua layer
for layer in base_model.layers:
    layer.trainable = False

# Custom classifier
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# CALLBACKS (INI YANG KITA TAMBAH 🔥)
# ======================

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Save model terbaik
checkpoint = ModelCheckpoint(
    filepath='D:\\Semester 6\\Deep Learning\\Projek Deni\\model\\model_terbaik.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Turunkan learning rate otomatis
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ======================
# TRAINING
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

# ======================
# SAVE FINAL MODEL
# ======================
model.save("D:\\Semester 6\\Deep Learning\\Projek Deni\\model\\model_final.h5")

# ======================
# VISUALISASI
# ======================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'val'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'val'])
plt.show()