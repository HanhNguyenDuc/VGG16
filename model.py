from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
IMG_SHAPE = X_train.shape[1:]

def model_vgg16():
    input_ = Input(shape = IMG_SHAPE)
    conv_1_1 = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(input_)
    conv_1_2 = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_1_1)
    norm_1 = BatchNormalization()(conv_1_2)
    drop_1 = Dropout(0.25)(norm_1)
    maxpool_1 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(drop_1)
    
    conv_2_1 = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(maxpool_1)
    conv_2_2 = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_2_1)
    norm_2 = BatchNormalization()(conv_2_2)
    drop_2 = Dropout(0.25)(norm_2)
    maxpool_2 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(drop_2)
    
    conv_3_1 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(maxpool_2)
    conv_3_2 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_3_1)
    conv_3_3 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_3_2)
    norm_3 = BatchNormalization()(conv_3_3)
    drop_3 = Dropout(0.25)(norm_3)
    maxpool_3 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(drop_3)
    
    conv_4_1 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(maxpool_3)
    conv_4_2 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_4_1)
    conv_4_3 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_4_2)
    norm_4 = BatchNormalization()(conv_4_3)
    drop_4 = Dropout(0.25)(norm_4)
    maxpool_4 = MaxPooling2D(pool_size =  (2, 2), padding = 'same')(drop_4)
    
    conv_5_1 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(maxpool_4)
    conv_5_2 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_5_1)
    conv_5_3 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_5_2)
    maxpool_5 = MaxPooling2D(pool_size = (2, 2))(conv_5_3)
    flatten_ = Flatten()(maxpool_5)
    dense_1 = Dense(4096)(flatten_)
    drop_ = Dropout(0.25)(dense_1)
    dense_2 = Dense(4096)(drop_)
    softmax_ = Dense(10, activation = 'softmax')(dense_2)

    return Model(inputs = input_, outputs = softmax_)

model = model_vgg16()
model.summary()

X_train = X_train / 255
# y_train = y_train / 255
X_test = X_test / 255
X_train, y_train = shuffle(X_train, y_train)

endp = int(0.9 * len(X_train))

X_val = X_train[endp:]
y_val = y_train[endp:]

X_train = X_train[:endp]
y_train = y_train[:endp]

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

datagen = datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, y_train), validation_data = (X_val, y_val), steps_per_epoch = len(X_train) / 32, epochs = 2)

loss, acc = model.evaluate(X_test, y_test)
print(loss, acc)
