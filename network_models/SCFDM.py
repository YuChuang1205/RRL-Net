#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""
from .network_module import *
from keras.models import *
from keras.layers import *

def SCFDM():
    input1 = Input(shape=(64, 64, 1))
    input2 = Input(shape=(64, 64, 1))
    con_2 = Concatenate(axis=-2)([input1, input2])

    x = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(con_2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) #32×64

    x = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) #16×32

    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 8×16


    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 4×8


    x1 = Lambda(lambda x: x[:, :, 0:4, :])(x)
    x2 = Lambda(lambda x: x[:, :, 4:8, :])(x)


    x1_pooling = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x1)
    x1_pooling = Lambda(lambda x: K.squeeze(x, axis=-2))(x1_pooling)
    x2_pooling = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x2)
    x2_pooling = Lambda(lambda x: K.squeeze(x, axis=-2))(x2_pooling)



    sim_out = Lambda(my_cosine_proximity)([x1_pooling,x2_pooling])
    r_diff1 = Subtract()([x1,x2])
    r_diff1 = Lambda(lambda x: abs(x))(r_diff1)
    res_fc0 = Flatten()(r_diff1)
    res_fc1 = Dense(1024, activation='relu')(res_fc0)
    res_fc2 = Dense(128, activation='relu')(res_fc1)
    res_fc3 = Dense(2, activation='softmax')(res_fc2)


    class_models = Model(inputs=[input1,input2], outputs=[res_fc3,sim_out])
    return class_models