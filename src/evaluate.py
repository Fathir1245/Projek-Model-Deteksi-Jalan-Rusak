import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("D:\\Semester 6\\Deep Learning\\Projek Deni\\model\\model_final.h5")

# ======================
# DATASET PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "road_surface_dataset")

# ======================
# DATA VALIDATION
# ======================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # PENTING!
)

# ======================
# PREDIKSI
# ======================
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_labels = list(val_data.class_indices.keys())

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ======================
# CLASSIFICATION REPORT
# ======================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))