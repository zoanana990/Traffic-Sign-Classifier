import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils
from ghostNet import GhostNet
from tensorflow.keras.preprocessing.image import array_to_img


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


### hyperparameter
size=(32, 32)
BATCH_SIZE = 25
EPOCHS = 20
CLASSES = 43
WEIGHTS_FINAL = 'GhostNet.h5'
FREEZE_LAYERS = 2
NUM_CLASSES = 43

### label
SignNames = pd.read_csv('signnames.csv').values
SignNames = np.array(SignNames)
SignNames = np.reshape(SignNames, (43, 2))
print("SignNames = \n", SignNames)
print('===============================================================================================================')

### classes we get
classes = SignNames[:, 1]
print("classes = \n", classes)
print('===============================================================================================================')

### load model
model = GhostNet((32, 32, 3), 43).build()
model.load_weights(WEIGHTS_FINAL)

### test data
### LOAD DATA
### data file path
testing_file = "data/test.p"

### open those data file
## mode rb means read by bytes
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

### split those file to image features and image label
X_test, y_test = test['features'], test['labels']
X_test_normalized = X_test / 255
y_test_onehot = utils.to_categorical(y_test)
X_test = np.array(X_test_normalized)
y_test = np.array(y_test_onehot)

print(len(X_test))

### prediction
index = np.random.randint(1, 12630, size=1, dtype=np.int32)
image = X_test[index]
image = np.reshape(image, (32, 32, 3))
image_show = array_to_img(image)
img = image.reshape(-1, 32, 32, 3)
image = img.reshape(32, 32, 3)
image = np.int32(image*255)

pred = model.predict(img)
print('This photo is a {}, predicted as a {}'.format(classes[np.argmax(y_test[index])], classes[np.argmax(pred)]))
plt.imshow(image)
plt.title('This photo is a {}, predicted as a {}'.format(classes[np.argmax(y_test[index])], classes[np.argmax(pred)]))
plt.xticks([])
plt.yticks([])
plt.show()







