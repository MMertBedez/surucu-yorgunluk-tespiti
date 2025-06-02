import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

def prepare_dataset(base_dir="open_closed_eyes_dataset", target_size=(224, 224)):

    open_eyes = []
    closed_eyes = []
    
    # Açık göz resimlerini yükle
    open_dir = os.path.join(base_dir, 'open')
    for img_name in os.listdir(open_dir):
        img_path = os.path.join(open_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:  # Geçersiz resimleri atla
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            open_eyes.append(img)
    
    # Kapalı göz resimlerini yükle
    closed_dir = os.path.join(base_dir, 'closed')
    for img_name in os.listdir(closed_dir):
        img_path = os.path.join(closed_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:  # Geçersiz resimleri atla
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            closed_eyes.append(img)
    
    print(f"Yüklenen açık göz sayısı: {len(open_eyes)}")
    print(f"Yüklenen kapalı göz sayısı: {len(closed_eyes)}")
    
    # Veri setlerini oluştur
    X = np.array(open_eyes + closed_eyes)
    y = np.array([1] * len(open_eyes) + [0] * len(closed_eyes))
    
    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

def visualize_samples(X, y, num_samples=5):
    """
    Örnek görüntüleri görselleştir
    """
    plt.figure(figsize=(15, 3))
    for i in range(min(num_samples, len(X))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title('Açık' if y[i] == 1 else 'Kapalı')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):

    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_eye_detection_model(input_shape=(224, 224, 3)):
    # VGG16 modelini yükle (weights='imagenet' ile önceden eğitilmiş ağırlıkları kullan)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Transfer learning için VGG16'nın son katmanlarını dondur
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Model mimarisini oluştur
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification için sigmoid
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
    
    # Veri artırma
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Callbacks tanımla
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=3e-5,
            verbose=1
        ),

    ]
    
    # Modeli derle
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Modeli eğit
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=validation_datagen.flow(X_val, y_val, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=len(X_train) // batch_size,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks
    )
    
    return history

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    # Veri setini hazırla
    print("Veri seti hazırlanıyor...")
    X_train, X_val, y_train, y_val = prepare_dataset()
    
    # Örnek görüntüleri görselleştir
    print("\nÖrnek görüntüler:")
    visualize_samples(X_train, y_train)
    
    # Model oluştur
    print("\nModel oluşturuluyor...")git init
    model = create_eye_detection_model()
    model.summary()
    
    # Modeli eğit
    print("\nModel eğitimi başlıyor...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Eğitim geçmişini görselleştir
    print("\nEğitim geçmişi:")
    plot_training_history(history)
    
    # Modeli kaydet
    save_model(model, 'eye_detection_model.h5')
    print("\nModel kaydedildi: eye_detection_model.h5")
