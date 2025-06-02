import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from  sklearn.model_selection import train_test_split
import seaborn as sns
from keras.models import Sequential
from tensorflow.keras.layers import Dense,  Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from  sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight


tf.config.list_physical_devices('GPU')

print("Available GPUs:", tf.config.list_physical_devices('GPU'))

print(tf.__version__)


path = r"C:\Users\pekta\Desktop\CALISMALAR\python\python\DersProje\yawn"

# Klasör isimlerini al ve sıralı bir şekilde label'lara çevir
label_names = sorted(os.listdir(path))  # ['no_yawn', 'yawn']
label_to_index = {name: idx for idx, name in enumerate(label_names)}
print("Label mapping:", label_to_index)

numOfClasses = len(label_names)
print("numOfClasses: ",numOfClasses)

images = []
classes = []

# Her klasörü gez, resimleri yükle, boyutlandır, etiketle
for label_name in label_names:
    folder_path = os.path.join(path, label_name)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            images.append(img)
            classes.append(label_to_index[label_name])
        else:
            print(f"Warning: Could not read image {image_path}")

# Verileri numpy dizisine çevir
images = np.array(images)
classes = np.array(classes)

print("images.shape:", images.shape)
print("classes.shape:", classes.shape)



x_train,x_test,y_train,y_test = train_test_split(images,classes,test_size=0.5,random_state=42)


x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.2,random_state=42)
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)
print("x_test.shape: ",x_test.shape)
print("y_test.shape: ",y_test.shape)
print("x_validation.shape: ",x_validation.shape)
print("y_validation.shape: ",y_validation.shape)




fig, ax = plt.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sns.countplot(y_train, ax = ax[0])
ax[0].set_title('y_train')

sns.countplot(y_test, ax = ax[1])
ax[1].set_title('y_test')

sns.countplot(y_validation, ax = ax[2])
ax[2].set_title('y_validation')


# preprocess
def PreProcessing(image):
    img =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img= cv2.equalizeHist(img)
    img = img/255
    return img


"""
idx = 50
img = PreProcessing(x_train[idx])
img = cv2.resize(img,(300,300))
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

x_train = np.array(list(map(PreProcessing,x_train)))
x_test = np.array(list(map(PreProcessing,x_test)))
x_validation = np.array(list(map(PreProcessing,x_validation)))


x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)


data_gen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)
data_gen.fit(x_train)

y_train = to_categorical(y_train,numOfClasses)
y_test = to_categorical(y_test,numOfClasses)
y_validation = to_categorical(y_validation,numOfClasses)



y_train_labels = np.argmax(y_train, axis=1)

# Sınıf ağırlıklarını hesapla
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),  # Sınıf etiketleri
    y=y_train_labels                   # Gerçek sınıf etiketleri
)

# Sınıf ağırlıklarını bir sözlüğe dönüştür
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)


# create model

model = Sequential()

model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(32, 32, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(units=numOfClasses, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 32

# ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,min_lr=0.0001)

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Modeli eğitirken tüm geri çağırma fonksiyonlarını kullan
hist = model.fit(
    data_gen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_validation, y_validation),
    epochs=100,
    verbose=1,
    steps_per_epoch=x_train.shape[0] // batch_size,
    shuffle=True,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

model.save("model_yawn.h5")



print("hist.history",hist.history)

plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.plot(hist.history['accuracy'],label='accuracy')
plt.plot(hist.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()


score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:',score[0])
print('Test accuracy:',score[1])



y_pred = model.predict(x_validation )

y_pred_class = np.argmax(y_pred, axis=1)

y_true =np.argmax(y_validation, axis=1)

cm= confusion_matrix(y_true,y_pred_class)

f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm,annot=True,ax=ax,linewidths=.01,fmt='.2%',linecolor="gray",cmap= "Greens")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.show()









































