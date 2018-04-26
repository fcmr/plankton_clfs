import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

#from skimage.transform import resize

## Generate dummy data
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

#Importing data
imagestrain_path = ('images_train.npy')
labelstrain_path = ('extended_labels_train.npy')
imagestest_path = ('images_test.npy')
labelstest_path = ('extended_labels_test.npy')

x_train = np.load(imagestrain_path)
y_train = np.load(labelstrain_path)
x_test = np.load(imagestest_path)
y_test = np.load(labelstest_path)

"""
#Resizing data
target_shape = (100, 100, 1)
new_imgs = np.zeros((len(x_train), target_shape[0], target_shape[1]))
for i in range(len(x_train)):
	new_imgs[i] = resize(x_train[i], (target_shape[0],target_shape[1])).astype('float32')
new_shape = (len(x_train), target_shape[0], target_shape[1], target_shape[2])
x_train = np.reshape(new_imgs, new_shape)

new_imgs1 = np.zeros((len(x_test), target_shape[0], target_shape[1]))
for j in range(len(x_test)):
	new_imgs1[j] = resize(x_test[j], (target_shape[0],target_shape[1])).astype('float32')
new_shape1 = (len(x_test), target_shape[0], target_shape[1], target_shape[2])
x_test = np.reshape(new_imgs1, new_shape1)
"""

#Defining the model
model = Sequential()
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#Training and evaluating
model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

print(score)
