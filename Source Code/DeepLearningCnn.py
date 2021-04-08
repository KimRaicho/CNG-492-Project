import os
import gc
import cv2
import random
from collections import defaultdict 
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# 1.) Get image path set

classes = defaultdict(list)  
for filename in os.listdir("CroppedImages"):

    userId = filename.split('_')[0]    # split the filename and retain only the first element which contains identity
    listname = "user_" + userId[1:]    # listname = user + identity without 'u' character
    classes[listname].append('CroppedImages/' + filename)


# 2.) Shuffling, Training and Testing Image Sets
train_set = []
test_set = []

for key in classes:
    random.shuffle(classes[key])  # shuffle so there is no  bias between training & testing set selection
    train_set.append(classes[key][:(len(classes[key])//2)])  # we use half as our training set
    test_set.append(classes[key][(len(classes[key])//2):])   # the other half as our test set

# un-list the sub-lists to make only one big list
train_set = list(chain.from_iterable(train_set))
test_set = list(chain.from_iterable(test_set))

# 4.) Garbage Collection
del classes
gc.collect()

# 5.) Image Pre-Processing - reformat images so that model knows what dimensions to expect
nRows = 150  # Width
nCols = 150  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1

# 6.) Training and Testing Set Labeling - we are going to use these arrays to contain the read images along with their label.
X_train = []
X_test = []
y_train = []
y_test = []

print(train_set)
# 7.) Read and Label Each Image in the Training Set
for image in train_set:
    try:
        # read in and resize the image based on our static dimensions from 5 above
        X_train.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))

        # to determine the class of each read image.
        if 'u014' in image:
            y_train.append(14)
        elif 'u029' in image:
            y_train.append(29)
        elif 'u033' in image:
            y_train.append(33)
        elif 'u051' in image:
            y_train.append(51)
        elif 'u057' in image:
            y_train.append(57)
        elif 'u075' in image:
            y_train.append(75)
        elif 'u088' in image:
            y_train.append(88)
        elif 'u102' in image:
            y_train.append(102)
        elif 'u112' in image:
            y_train.append(112)
        elif 'u123' in image:
            y_train.append(123)
        elif 'u133' in image:
            y_train.append(133)
        elif 'u154' in image:
            y_train.append(154)

    except Exception:   # images failed to be read
        print('Failed to format: ', image)


# 8.) Read and Label Each Image in the Testing Set
for image in test_set:
    try:
        # read in and resize the image based on our static dimensions from 5 above
        X_test.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))

        # to determine the class of each read image.
        if 'u014' in image:
            y_test.append(14)
        elif 'u029' in image:
            y_test.append(29)
        elif 'u033' in image:
            y_test.append(33)
        elif 'u051' in image:
            y_test.append(51)
        elif 'u057' in image:
            y_test.append(57)
        elif 'u075' in image:
            y_test.append(75)
        elif 'u088' in image:
            y_test.append(88)
        elif 'u102' in image:
            y_test.append(102)
        elif 'u112' in image:
            y_test.append(112)
        elif 'u123' in image:
            y_test.append(123)
        elif 'u133' in image:
            y_test.append(133)
        elif 'u154' in image:
            y_test.append(154)

    except Exception:   # images failed to be read
        print('Failed to format: ', image)


# 9.) Garbage Collection
del train_set, test_set
gc.collect()


# 10.) Convert to Numpy Arrays so that it can be fed to CNN, and reformat the input and target data using accompanying libraries like Scikit-learn and Keras.
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# 11.) Switch Targets to Categorical - So that we can use a softmax activation function X ∈ [0, 1] to predict the image class we are going to convert our vector of labels where L ∈ {14 - 154} to a categorical set L ∈ {0, 1}.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 12.) Convolutional Neural Network
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))   # learn 32 filters, filter size = 3*3, activation(get output of node) used =  relu(Rectified linear unit), used to determine output of neural network (btwn 0-1)
model.add(MaxPooling2D(2, 2))    # down-sample input
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())   # prepares a vector for the fully connected layers
model.add(Dropout(0.1))   #
model.add(Dense(155, activation='softmax'))  # Full connected layer the results of the convolutional layers are fed through one or more neural layers to generate a prediction.
#model.add(Dense(10, activation='softmax'))


# 13.) Model Summary
print(model.summary())

# 14.) Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)

# 15.) Plot Accuracy Over Training Period
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 16.) Plot Loss Over Training Period
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
