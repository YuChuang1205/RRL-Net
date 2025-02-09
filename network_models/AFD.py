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

    conv1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='conv1')(input_img)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = ReLU(name='relu1')(bn1)


    conv2 = Convolution2D(32, (3, 3),  strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv2')(relu1)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = ReLU(name='relu2')(bn2)
    in1 = InstanceNormalization(name='in1')(relu2)
    relu3 = ReLU(name='relu3')(in1)


    conv3 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv3')(relu3)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu4 = ReLU(name='relu4')(bn3)
    conv4 = Convolution2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv4')(relu4)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu5 = ReLU(name='relu5')(bn4)
    in2 = InstanceNormalization(name='in2')(relu5)
    relu6 = ReLU(name='relu6')(in2)


    conv5 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='conv5')(relu6)
    bn5 = BatchNormalization(name='bn5')(conv5)
    relu7 = ReLU(name='relu7')(bn5)
    conv6 = Convolution2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv6')(relu7)
    bn6 = BatchNormalization(name='bn6')(conv6)
    relu8 = ReLU(name='relu8')(bn6)


    conv7 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='conv7')(relu8)
    bn7 = BatchNormalization(name='bn7')(conv7)
    relu9 = ReLU(name='relu9')(bn7)
    conv8 = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv8')(relu9)
    bn8 = BatchNormalization(name='bn8')(conv8)
    relu10 = ReLU(name='relu10')(bn8)
    model = Model(inputs=input_img, outputs=[relu1,relu3,relu6,relu8,relu10])
    return model

def AFD():
    model1 = FeatureNetwork2()
    model2 = FeatureNetwork2()

    for layer in model2.layers:
        layer.name = layer.name + str("_2")

    feature1_1, feature2_1, feature3_1, feature4_1, feature5_1 = model1.output
    feature1_2, feature2_2, feature3_2, feature4_2, feature5_2 = model2.output


    sub_1 = Subtract()([feature1_1, feature1_2])
    sub_1 = Lambda(lambda x: K.abs(x))(sub_1)
    sub_1 = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(sub_1)
    sub_1 = BatchNormalization()(sub_1)
    sub_1 = ReLU()(sub_1)
    sub_1 = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(sub_1)
    sub_1 = BatchNormalization()(sub_1)
    sub_1 = ReLU()(sub_1)


    sub_2 = Subtract()([feature2_1, feature2_2])
    sub_2 = Lambda(lambda x: K.abs(x))(sub_2)
    con_2 = Concatenate()([sub_1,sub_2])
    con_2 = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(con_2)
    con_2 = BatchNormalization()(con_2)
    con_2 = ReLU()(con_2)
    con_2 = Convolution2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con_2)
    con_2 = BatchNormalization()(con_2)
    con_2 = ReLU()(con_2)


    sub_3 = Subtract()([feature3_1,feature3_2])
    sub_3 = Lambda(lambda x: K.abs(x))(sub_3)
    con_3 = Concatenate()([con_2, sub_3])  # 16×16×128
    con_3 = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(con_3)
    con_3 = BatchNormalization()(con_3)
    con_3 = ReLU()(con_3)
    con_3 = Convolution2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con_3)
    con_3 = BatchNormalization()(con_3)
    con_3 = ReLU()(con_3)


    sub_4 = Subtract()([feature4_1, feature4_2])
    sub_4 = Lambda(lambda x: K.abs(x))(sub_4)
    con_4 = Concatenate()([con_3, sub_4])  # 8×8×256
    con_4 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(con_4)
    con_4 = BatchNormalization()(con_4)
    con_4 = ReLU()(con_4)
    con_4 = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con_4)
    con_4 = BatchNormalization()(con_4)
    con_4 = ReLU()(con_4)


    sub_5 = Subtract()([feature5_1, feature5_2])
    sub_5 = Lambda(lambda x: K.abs(x))(sub_5)
    con_5 = Concatenate()([con_4, sub_5])  # 4×4×512
    con_5 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(con_5)
    con_5 = BatchNormalization()(con_5)
    con_5 = ReLU()(con_5)
    con_5 = AveragePooling2D(pool_size=[4,4])(con_5)


    res_fc0 = Flatten()(con_5)
    res_fc1 = Dense(512, activation='relu')(res_fc0)
    res_fc2 = Dense(256, activation='relu')(res_fc1)
    res_fc3 = Dense(2, activation='softmax')(res_fc2)



    sub_5_single = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(sub_5)
    sub_5_single = BatchNormalization()(sub_5_single)
    sub_5_single = ReLU()(sub_5_single)
    sub_5_single = AveragePooling2D(pool_size=[4,4])(sub_5_single)
    res_s_fc0 = Flatten()(sub_5_single)
    res_s_fc1 = Dense(512, activation='relu')(res_s_fc0)
    res_s_fc2 = Dense(256, activation='relu')(res_s_fc1)
    res_s_fc3 = Dense(2, activation='softmax')(res_s_fc2)


    class_models = Model(inputs=[model1.input, model2.input], outputs=[res_fc3,res_s_fc3])
    return class_models





