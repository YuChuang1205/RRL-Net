#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""

from keras.models import *
from keras.layers import *
from .network_module import *

def FeatureNetwork2():
    input_img = Input(shape=(64, 64, 1))

    x = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = imp_eca(x, gamma=2, b=1)


    x = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = imp_eca(x, gamma=2, b=1)


    x = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = imp_eca(x, gamma=2, b=1)


    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = imp_eca(x, gamma=2, b=1)


    model = Model(inputs=input_img, outputs=x)
    return model

def EFR():
    model1 = FeatureNetwork2()
    model2 = FeatureNetwork2()
    model3 = FeatureNetwork2()
    model4 = FeatureNetwork2()

    for layer in model2.layers:
        layer.name = layer.name + str("_2")
    for layer in model3.layers:
        layer.name = layer.name + str("_3")
    for layer in model4.layers:
        layer.name = layer.name + str("_4")

    sub_2 = Subtract()([model1.output, model2.output])
    sub_2 = Lambda(lambda x: K.abs(x))(sub_2)

    sub_2 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(sub_2)
    sub_2 = BatchNormalization()(sub_2)
    sub_2 = LeakyReLU(0.1)(sub_2)


    c1con_2 = Multiply()([model3.output, model4.output])
    c1con_2 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c1con_2)
    c1con_2 = BatchNormalization()(c1con_2)
    c1con_2 = LeakyReLU(0.1)(c1con_2)


    con_2 = Concatenate()([sub_2,c1con_2])
    con_2 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(con_2)
    con_2 = BatchNormalization()(con_2)
    con_2 = LeakyReLU(0.1)(con_2)


    sub_fc0 = GlobalAveragePooling2D()(sub_2)
    sub_fc1 = Dense(512, activation='relu')(sub_fc0)
    sub_fc2 = Dense(256, activation='relu')(sub_fc1)
    sub_fc3 = Dense(1, activation='sigmoid')(sub_fc2)

    c1con_fc0 = GlobalAveragePooling2D()(c1con_2)
    c1con_fc1 = Dense(512, activation='relu')(c1con_fc0)
    c1con_fc2 = Dense(256, activation='relu')(c1con_fc1)
    c1con_fc3 = Dense(1, activation='sigmoid')(c1con_fc2)

    res_fc0 = GlobalAveragePooling2D()(con_2)
    res_fc1 = Dense(512, activation='relu')(res_fc0)
    res_fc2 = Dense(256, activation='relu')(res_fc1)
    res_fc3 = Dense(1, activation='sigmoid')(res_fc2)

    class_models = Model(inputs=[model1.input, model2.input, model3.input, model4.input], outputs=[res_fc3,sub_fc3,c1con_fc3])
    return class_models