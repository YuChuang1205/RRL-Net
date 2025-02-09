#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""

from keras.models import *
from keras.layers import *


def FeatureNetwork2():
    input_img = Input(shape=(64, 64, 1))

    conv1_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)
    conv1_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)


    conv2_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    conv2_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2_1)
    conv2_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    conv2_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)


    conv1_1_pooling = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_1_pooling = BatchNormalization()(conv1_1_pooling)
    conv1_1_pooling = LeakyReLU(0.1)(conv1_1_pooling)
    conv1_1_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1_1_pooling)
    conv2_1_add = Add()([conv2_1,conv1_1_pooling])
    conv2_1_add = LeakyReLU(0.1)(conv2_1_add)


    conv3_1 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv2_1_add)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    conv3_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3_1)
    conv3_1 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    conv3_1 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)


    conv2_1_pooling = Convolution2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(conv2_1_add)
    conv2_1_pooling = BatchNormalization()(conv2_1_pooling)
    conv2_1_pooling = LeakyReLU(0.1)(conv2_1_pooling)  # 64×64
    conv2_1_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2_1_pooling)
    conv3_1_add = Add()([conv3_1, conv2_1_pooling])
    conv3_1_add = LeakyReLU(0.1)(conv3_1_add)


    conv4_1 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv3_1_add)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    conv4_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4_1)
    conv4_1 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv4_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    conv4_1 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv4_1)
    conv4_1 = BatchNormalization()(conv4_1)


    conv3_1_pooling = Convolution2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(conv3_1_add)
    conv3_1_pooling = BatchNormalization()(conv3_1_pooling)
    conv3_1_pooling = LeakyReLU(0.1)(conv3_1_pooling)
    conv3_1_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3_1_pooling)
    conv4_1_add = Add()([conv4_1, conv3_1_pooling])
    conv4_1_add = LeakyReLU(0.1)(conv4_1_add)



    conv5_1 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv4_1_add)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    conv5_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv5_1)
    conv5_1 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv5_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    conv5_1 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv5_1)
    conv5_1 = BatchNormalization()(conv5_1)


    conv4_1_pooling = Convolution2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(conv4_1_add)
    conv4_1_pooling = BatchNormalization()(conv4_1_pooling)
    conv4_1_pooling = LeakyReLU(0.1)(conv4_1_pooling)
    conv4_1_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4_1_pooling)
    conv5_1_add = Add()([conv5_1, conv4_1_pooling])
    conv5_1_add = LeakyReLU(0.1)(conv5_1_add)


    model = Model(inputs=input_img, outputs=[conv1_1, conv2_1_add, conv3_1_add, conv4_1_add, conv5_1_add])
    return model

def FIL():
    model1 = FeatureNetwork2()
    model2 = FeatureNetwork2()

    for layer in model2.layers:
        layer.name = layer.name + str("_2")

    feature1_1, feature2_1, feature3_1, feature4_1, feature5_1 = model1.output
    feature1_2, feature2_2, feature3_2, feature4_2, feature5_2 = model2.output


    a_feature1_1 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature1_1)
    a_feature1_1 = BatchNormalization()(a_feature1_1)
    a_feature1_1 = LeakyReLU(0.1)(a_feature1_1)

    as_feature1_1 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature1_1)
    as_feature1_1 = BatchNormalization()(as_feature1_1)
    as_feature1_1 = Lambda(lambda x: K.sigmoid(x))(as_feature1_1)

    a_feature1_2 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature1_2)
    a_feature1_2 = BatchNormalization()(a_feature1_2)
    a_feature1_2 = LeakyReLU(0.1)(a_feature1_2)

    as_feature1_2 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature1_2)
    as_feature1_2 = BatchNormalization()(as_feature1_2)
    as_feature1_2 = Lambda(lambda x: K.sigmoid(x))(as_feature1_2)

    a_feature1_3 = Lambda(lambda x: x[0] * x[1])([a_feature1_1, as_feature1_2])
    a_feature1_4 = Lambda(lambda x: x[0] * x[1])([a_feature1_2, as_feature1_1])

    as_feature1_3 = Subtract()([a_feature1_3, a_feature1_1])
    as_feature1_3 = Lambda(lambda x: K.abs(x))(as_feature1_3)
    as_feature1_4 = Subtract()([a_feature1_4, a_feature1_2])
    as_feature1_4 = Lambda(lambda x: K.abs(x))(as_feature1_4)


    a_feature2_1 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature2_1)
    a_feature2_1 = BatchNormalization()(a_feature2_1)
    a_feature2_1 = LeakyReLU(0.1)(a_feature2_1)

    as_feature2_1 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature2_1)
    as_feature2_1 = BatchNormalization()(as_feature2_1)
    as_feature2_1 = Lambda(lambda x: K.sigmoid(x))(as_feature2_1)

    a_feature2_2 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature2_2)
    a_feature2_2 = BatchNormalization()(a_feature2_2)
    a_feature2_2 = LeakyReLU(0.1)(a_feature2_2)

    as_feature2_2 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(feature2_2)
    as_feature2_2 = BatchNormalization()(as_feature2_2)
    as_feature2_2 = Lambda(lambda x: K.sigmoid(x))(as_feature2_2)

    a_feature2_3 = Lambda(lambda x: x[0] * x[1])([a_feature2_1, as_feature2_2])
    a_feature2_4 = Lambda(lambda x: x[0] * x[1])([a_feature2_2, as_feature2_1])
    as_feature2_3 = Subtract()([a_feature2_3, a_feature2_1])
    as_feature2_3 = Lambda(lambda x: K.abs(x))(as_feature2_3)
    as_feature2_4 = Subtract()([a_feature2_4, a_feature2_2])
    as_feature2_4 = Lambda(lambda x: K.abs(x))(as_feature2_4)


    a_feature3_1 = Convolution2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(feature3_1)
    a_feature3_1 = BatchNormalization()(a_feature3_1)
    a_feature3_1 = LeakyReLU(0.1)(a_feature3_1)

    as_feature3_1 = Convolution2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(feature3_1)
    as_feature3_1 = BatchNormalization()(as_feature3_1)
    as_feature3_1 = Lambda(lambda x: K.sigmoid(x))(as_feature3_1)

    a_feature3_2 = Convolution2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(feature3_2)
    a_feature3_2 = BatchNormalization()(a_feature3_2)
    a_feature3_2 = LeakyReLU(0.1)(a_feature3_2)

    as_feature3_2 = Convolution2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(feature3_2)
    as_feature3_2 = BatchNormalization()(as_feature3_2)
    as_feature3_2 = Lambda(lambda x: K.sigmoid(x))(as_feature3_2)

    a_feature3_3 = Lambda(lambda x: x[0] * x[1])([a_feature3_1, as_feature3_2])
    a_feature3_4 = Lambda(lambda x: x[0] * x[1])([a_feature3_2, as_feature3_1])
    as_feature3_3 = Subtract()([a_feature3_3, a_feature3_1])
    as_feature3_3 = Lambda(lambda x: K.abs(x))(as_feature3_3)
    as_feature3_4 = Subtract()([a_feature3_4, a_feature3_2])
    as_feature3_4 = Lambda(lambda x: K.abs(x))(as_feature3_4)


    a_feature4_1 = Convolution2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(feature4_1)
    a_feature4_1 = BatchNormalization()(a_feature4_1)
    a_feature4_1 = LeakyReLU(0.1)(a_feature4_1)

    as_feature4_1 = Convolution2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(feature4_1)
    as_feature4_1 = BatchNormalization()(as_feature4_1)
    as_feature4_1 = Lambda(lambda x: K.sigmoid(x))(as_feature4_1)

    a_feature4_2 = Convolution2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(feature4_2)
    a_feature4_2 = BatchNormalization()(a_feature4_2)
    a_feature4_2 = LeakyReLU(0.1)(a_feature4_2)

    as_feature4_2 = Convolution2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(feature4_2)
    as_feature4_2 = BatchNormalization()(as_feature4_2)
    as_feature4_2 = Lambda(lambda x: K.sigmoid(x))(as_feature4_2)

    a_feature4_3 = Lambda(lambda x: x[0] * x[1])([a_feature4_1, as_feature4_2])
    a_feature4_4 = Lambda(lambda x: x[0] * x[1])([a_feature4_2, as_feature4_1])
    as_feature4_3 = Subtract()([a_feature4_3, a_feature4_1])
    as_feature4_3 = Lambda(lambda x: K.abs(x))(as_feature4_3)
    as_feature4_4 = Subtract()([a_feature4_4, a_feature4_2])
    as_feature4_4 = Lambda(lambda x: K.abs(x))(as_feature4_4)


    a_feature5_1 = Convolution2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(feature5_1)
    a_feature5_1 = BatchNormalization()(a_feature5_1)
    a_feature5_1 = LeakyReLU(0.1)(a_feature5_1)

    as_feature5_1 = Convolution2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(feature5_1)
    as_feature5_1 = BatchNormalization()(as_feature5_1)
    as_feature5_1 = Lambda(lambda x: K.sigmoid(x))(as_feature5_1)

    a_feature5_2 = Convolution2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(feature5_2)
    a_feature5_2 = BatchNormalization()(a_feature5_2)
    a_feature5_2 = LeakyReLU(0.1)(a_feature5_2)

    as_feature5_2 = Convolution2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(feature5_2)
    as_feature5_2 = BatchNormalization()(as_feature5_2)
    as_feature5_2 = Lambda(lambda x: K.sigmoid(x))(as_feature5_2)

    a_feature5_3 = Lambda(lambda x: x[0] * x[1])([a_feature5_1, as_feature5_2])
    a_feature5_4 = Lambda(lambda x: x[0] * x[1])([a_feature5_2, as_feature5_1])
    as_feature5_3 = Subtract()([a_feature5_3, a_feature5_1])
    as_feature5_3 = Lambda(lambda x: K.abs(x))(as_feature5_3)
    as_feature5_4 = Subtract()([a_feature5_4, a_feature5_2])
    as_feature5_4 = Lambda(lambda x: K.abs(x))(as_feature5_4)


    ccon_1 = Concatenate()([as_feature1_3, as_feature1_4])
    con_1 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(ccon_1)
    con_1 = BatchNormalization()(con_1)
    con_1 = LeakyReLU(0.1)(con_1)
    con_1 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(con_1)
    con_1 = BatchNormalization()(con_1)
    con_1 = LeakyReLU(0.1)(con_1)  # 32×32×64
    mcon_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(con_1)


    ccon_2 = Concatenate()([as_feature2_3, as_feature2_4])
    con_2 = Concatenate()([mcon_1, ccon_2])
    con_2 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(con_2)
    con_2 = BatchNormalization()(con_2)
    con_2 = LeakyReLU(0.1)(con_2)
    con_2 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(con_2)
    con_2 = BatchNormalization()(con_2)
    con_2 = LeakyReLU(0.1)(con_2)  # 16×16×128
    mcon_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(con_2)


    ccon_3 = Concatenate()([as_feature3_3, as_feature3_4])
    con_3 = Concatenate()([mcon_2, ccon_3])
    con_3 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(con_3)
    con_3 = BatchNormalization()(con_3)
    con_3 = LeakyReLU(0.1)(con_3)
    con_3 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(con_3)
    con_3 = BatchNormalization()(con_3)
    con_3 = LeakyReLU(0.1)(con_3)  # 8×8×256
    mcon_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(con_3)


    ccon_4 = Concatenate()([as_feature4_3, as_feature4_4])
    con_4 = Concatenate()([mcon_3, ccon_4])
    con_4 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(con_4)
    con_4 = BatchNormalization()(con_4)
    con_4 = LeakyReLU(0.1)(con_4)
    con_4 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(con_4)
    con_4 = BatchNormalization()(con_4)
    con_4 = LeakyReLU(0.1)(con_4)  # 4×4×512
    mcon_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(con_4)


    ccon_5 = Concatenate()([as_feature5_3, as_feature5_4])
    con_5 = Concatenate()([mcon_4, ccon_5])
    con_5 = Convolution2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(con_5)
    con_5 = BatchNormalization()(con_5)
    con_5 = LeakyReLU(0.1)(con_5)
    con_5 = Convolution2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(con_5)
    con_5 = BatchNormalization()(con_5)
    con_5 = LeakyReLU(0.1)(con_5)  # 2×2×1024



    loss_3 = Add()([a_feature3_3, a_feature3_4])
    loss_3 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(loss_3)
    loss_3 = BatchNormalization()(loss_3)
    loss_3 = LeakyReLU(0.1)(loss_3)
    loss_3 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(loss_3)
    loss_3 = BatchNormalization()(loss_3)
    loss_3 = LeakyReLU(0.1)(loss_3)
    loss_3 = GlobalAveragePooling2D()(loss_3)
    loss_3 = Dense(512, activation='relu')(loss_3)
    loss_3 = Dense(256, activation='relu')(loss_3)
    loss_3 = Dense(1, activation='sigmoid')(loss_3)


    loss_4 = Add()([a_feature4_3, a_feature4_4])
    loss_4 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(loss_4)
    loss_4 = BatchNormalization()(loss_4)
    loss_4 = LeakyReLU(0.1)(loss_4)
    loss_4 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(loss_4)
    loss_4 = BatchNormalization()(loss_4)
    loss_4 = LeakyReLU(0.1)(loss_4)

    loss_4 = GlobalAveragePooling2D()(loss_4)
    loss_4 = Dense(512, activation='relu')(loss_4)
    loss_4 = Dense(256, activation='relu')(loss_4)
    loss_4 = Dense(1, activation='sigmoid')(loss_4)


    loss_5 = Add()([a_feature5_3, a_feature5_4])
    loss_5 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(loss_5)
    loss_5 = BatchNormalization()(loss_5)
    loss_5 = LeakyReLU(0.1)(loss_5)
    loss_5 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(loss_5)
    loss_5 = BatchNormalization()(loss_5)
    loss_5 = LeakyReLU(0.1)(loss_5)
    loss_5 = GlobalAveragePooling2D()(loss_5)
    loss_5 = Dense(512, activation='relu')(loss_5)
    loss_5 = Dense(256, activation='relu')(loss_5)
    loss_5 = Dense(1, activation='sigmoid')(loss_5)


    branch3_fc0 = GlobalAveragePooling2D()(con_3)
    branch3_fc1 = Dense(512, activation='relu')(branch3_fc0)
    branch3_fc2 = Dense(256, activation='relu')(branch3_fc1)
    branch3_fc3 = Dense(1, activation='sigmoid')(branch3_fc2)


    branch4_fc0 = GlobalAveragePooling2D()(con_4)
    branch4_fc1 = Dense(512, activation='relu')(branch4_fc0)
    branch4_fc2 = Dense(256, activation='relu')(branch4_fc1)
    branch4_fc3 = Dense(1, activation='sigmoid')(branch4_fc2)


    branch5_fc0 = GlobalAveragePooling2D()(con_5)
    branch5_fc1 = Dense(512, activation='relu')(branch5_fc0)
    branch5_fc2 = Dense(256, activation='relu')(branch5_fc1)
    branch5_fc3 = Dense(1, activation='sigmoid')(branch5_fc2)



    class_models = Model(inputs=[model1.input, model2.input], outputs=[branch5_fc3, branch4_fc3,branch3_fc3, loss_5, loss_4, loss_3])
    return class_models