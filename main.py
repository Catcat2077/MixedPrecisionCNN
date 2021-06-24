import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras import Model
# from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(threshold=np.inf)       #输出无限制

cifar10 = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# 数据预处理
def normalization(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))     # 批数 像素x像素 通道数
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype(np.float32)  # 数据类型转换
    x_test = x_test.astype(np.float32)

    (x_train, x_test) = normalization(x_train, x_test)

    y_train = to_categorical(y_train, 10)  # one-hot
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()
# 图像增强

image_gen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
image_gen_train.fit(x_train)



class VGG_conv(Model):
    def __init__(self, filters, kernel_size, padding, weight_decay):
        super(VGG_conv, self).__init__()             #继承父类初始化

        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           kernel_regularizer=regularizers.l2(weight_decay))
        self.bn = BatchNormalization()
        self.act = Activation('relu')


    def call(self, l):
        l = self.conv(l)
        l = self.bn(l)
        l = self.act(l)

        return l


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()

        # convolutional layer 1
        self.c1 = VGG_conv(filters=64, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout1 = Dropout(0.4)
        self.c2 = VGG_conv(filters=64, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.pooling1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # convolutional layer 2
        self.c3 = VGG_conv(filters=128, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout2 = Dropout(0.4)
        self.c4 = VGG_conv(filters=128, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.pooling2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # convolutional layer 3
        self.c5 = VGG_conv(filters=256, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout3 = Dropout(0.4)
        self.c6 = VGG_conv(filters=256, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout4 = Dropout(0.4)
        self.c7 = VGG_conv(filters=256, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.pooling3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # convolutional layer 4
        self.c8 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout5 = Dropout(0.4)
        self.c9 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout6 = Dropout(0.4)
        self.c10 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.pooling4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # convolutional layer 5
        self.c11 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout7 = Dropout(0.4)
        self.c12 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.dropout8 = Dropout(0.4)
        self.c13 = VGG_conv(filters=512, kernel_size=[3, 3], padding='same', weight_decay=0.0005)
        self.pooling5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # full connected layer
        self.flatten = Flatten()
        self.dropout9 = Dropout(0.5)
        self.f1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.f2 = Dense(10, activation='softmax')


    def call(self, x):                  #调用VGG块
        x = self.c1(x)
        x = self.dropout1(x)
        x = self.c2(x)
        x = self.pooling1(x)

        x = self.c3(x)
        x = self.dropout2(x)
        x = self.c4(x)
        x = self.pooling2(x)

        x = self.c5(x)
        x = self.dropout3(x)
        x = self.c6(x)
        x = self.dropout4(x)
        x = self.c7(x)
        x = self.pooling3(x)

        x = self.c8(x)
        x = self.dropout5(x)
        x = self.c9(x)
        x = self.dropout6(x)
        x = self.c10(x)
        x = self.pooling4(x)

        x = self.c11(x)
        x = self.dropout7(x)
        x = self.c12(x)
        x = self.dropout8(x)
        x = self.c13(x)
        x = self.pooling5(x)

        x = self.flatten(x)
        x = self.dropout9(x)
        x = self.f1(x)
        y = self.f2(x)

        return y                #返回

model = VGG16()

# opt = optimizers.Adam(learning_rate=0.001)

lr = 0.1
lr_drop = 20

def lr_scheduler(epoch):
    return lr * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

opt = tf.keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_save_path = "./checkpoint/VGG.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------------------------load weights------------------------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True, save_weights_only=True)

# 训练参数
batch_size = 128
epochs = 250


history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    verbose=2,
                    callbacks=[cp_callback, reduce_lr],
                    validation_data=(x_test, y_test),
                    validation_freq=1)


'''
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1,
                    callbacks=[cp_callback, reduce_lr],
                    validation_data=(x_test, y_test),
                    validation_freq=1)
'''

model.summary()

# 写入数据
#print(model.trainable_variables)
# file = open('./VGGweights.txt','w')
# for i in model.trainable_variables:
#     file.write((str(i.name) + '\n'))
#     file.write((str(i.shape) + '\n'))
#     file.write((str(i.numpy()) + '\n'))
# file.close()
model.save('.\params')

# 绘图
accuracy = history.history['accuracy']
t_accuracy = history.history['val_accuracy']
loss = history.history['loss']
t_loss = history.history['val_loss']
plt.plot(accuracy, label='Accuracy')
plt.plot(t_accuracy, label='Test Accuracy')
plt.title(label='Accuracy')
plt.legend()
plt.show()

# plt.plot(loss, label='Loss')
# plt.plot(t_loss, label='Test Loss')
# plt.title(label='Loss')
# plt.legend()
# plt.show()