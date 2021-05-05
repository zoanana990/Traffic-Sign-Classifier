import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils as utils
import ghostNet
from numba import cuda

### release all gpu memory
device = cuda.get_current_device()
device.reset()

### GPU Activation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

### LOAD DATA
### data file path
training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"

### open those data file
## mode rb means read by bytes
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

### split those file to image features and image label
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Data Summary
"""
The pickled data is a dictionary with 4 key/value pairs:

'features' is a 4D array containing raw pixel data of the traffic sign images, 
(num examples, width, height, channels).
'labels' is a 2D array containing the label/class id of the traffic sign. 
The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
'coords' is a list containing tuples, 
(x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 
THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
"""
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)

## the shape of an traffic sign image
image_shape = X_train[0].shape[:-1]

## number of classes in the dataset
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print('===============================================================================================================')

## DATA EXPLORATION VISUALIZATION
fig, ax = plt.subplots()
ax.bar(range(n_classes), np.bincount(y_train), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The Count of each Signs')
plt.show()

SignNames = pd.read_csv('signnames.csv').values
print("SignNames = \n", SignNames)
print('===============================================================================================================')
classes = SignNames[:, 1]

plt.figure(figsize=(16, 16))
for c in range(n_classes):
    i = np.random.choice(np.where(y_train == c)[0])
    plt.subplot(11, 4, c+1)
    plt.axis('off')
    plt.title('class:{}'.format(classes[c]))
    plt.imshow(X_train[i])
plt.show()

## Normalize the train and test datasets to (-1,1)
X_train_normalized = X_train / 255
X_valid_normalized = X_valid / 255
X_test_normalized = X_test / 255
y_train_onehot = utils.to_categorical(y_train)
y_valid_onehot = utils.to_categorical(y_valid)
y_test_onehot = utils.to_categorical(y_test)

# print the result
print("X_train.shape = ", X_train_normalized.shape)
print("y_train.shape = ", y_train_onehot.shape)
print("X_valid.shape = ", X_valid_normalized.shape)
print("y_valid.shape = ", y_valid_onehot.shape)
print("X_test.shape = ", X_test_normalized.shape)
print("y_test.shape = ", y_test_onehot.shape)
print('===============================================================================================================')
print("Normalize Data Finished")
print('===============================================================================================================')

### convert the tuple data to array
X_train = np.array(X_train_normalized)
y_train = np.array(y_train_onehot)
X_valid = np.array(X_valid_normalized)
y_valid = np.array(y_valid_onehot)
X_test = np.array(X_test_normalized)
y_test = np.array(y_test_onehot)


### hyperparameter
size=(32, 32)
BATCH_SIZE = 25
EPOCHS = 20
CLASSES = 43
WEIGHTS_FINAL = 'GhostNet.h5'
FREEZE_LAYERS = 2
NUM_CLASSES = 43

### training model
model = ghostNet.GhostNet((32, 32, 3), NUM_CLASSES).build()
model.summary()

### use pretraining model
# try:
#     model.load_weights(WEIGHTS_FINAL)
# except:
#     pass
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
result = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                   validation_data=(X_valid, y_valid), shuffle=True)
model.save_weights(WEIGHTS_FINAL)

### plot
plt.figure()
plt.plot(result.epoch, result.history['acc'], label="acc")
plt.plot(result.epoch, result.history['val_acc'], label="val_acc")
plt.scatter(result.epoch, result.history['acc'], marker='*')
plt.scatter(result.epoch, result.history['val_acc'])
plt.legend(loc='lower right')
plt.title("GhostNet Accuracy")
plt.show()

plt.figure()
plt.plot(result.epoch, result.history['loss'], label="loss")
plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
plt.scatter(result.epoch, result.history['loss'], marker='*')
plt.scatter(result.epoch, result.history['val_loss'])
plt.legend(loc='upper right')
plt.title("GhostNet Loss")
plt.show()

