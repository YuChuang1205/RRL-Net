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

    x = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='conv1')(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling1')(x)

    x = Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='conv3')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling2')(x)

    x = Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='conv4')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling3')(x)

    x = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='conv5')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling4')(x)

    model = Model(inputs=input_img, outputs=x)
    return model



def MFD():
    model1 = FeatureNetwork2()
    model2 = FeatureNetwork2()
    model3 = FeatureNetwork2()
    model4 = FeatureNetwork2()
    model5 = FeatureNetwork2()
    model6 = FeatureNetwork2()
    model7 = FeatureNetwork2()
    model8 = FeatureNetwork2()

    for layer in model2.layers:
        layer.name = layer.name + str("_2")

    for layer in model3.layers:
        layer.name = layer.name + str("_3")

    for layer in model4.layers:
        layer.name = layer.name + str("_4")

    for layer in model5.layers:
        layer.name = layer.name + str("_5")

    for layer in model6.layers:
        layer.name = layer.name + str("_6")

    for layer in model7.layers:
        layer.name = layer.name + str("_7")

    for layer in model8.layers:
        layer.name = layer.name + str("_8")

    conv1_1out = model1.get_layer('conv1').output
    conv1_2out = model2.get_layer('conv1_2').output
    conv1_3out = model3.get_layer('conv1_3').output
    conv1_4out = model4.get_layer('conv1_4').output
    conv1_5out = model5.get_layer('conv1_5').output
    conv1_6out = model6.get_layer('conv1_6').output
    conv1_7out = model7.get_layer('conv1_7').output
    conv1_8out = model8.get_layer('conv1_8').output

    conv2_1out = model1.get_layer('conv2').output
    conv2_2out = model2.get_layer('conv2_2').output
    conv2_3out = model3.get_layer('conv2_3').output
    conv2_4out = model4.get_layer('conv2_4').output
    conv2_5out = model5.get_layer('conv2_5').output
    conv2_6out = model6.get_layer('conv2_6').output
    conv2_7out = model7.get_layer('conv2_7').output
    conv2_8out = model8.get_layer('conv2_8').output

    pooling1_1out = model1.get_layer('pooling1').output
    pooling1_2out = model2.get_layer('pooling1_2').output
    pooling1_3out = model3.get_layer('pooling1_3').output
    pooling1_4out = model4.get_layer('pooling1_4').output
    pooling1_5out = model5.get_layer('pooling1_5').output
    pooling1_6out = model6.get_layer('pooling1_6').output
    pooling1_7out = model7.get_layer('pooling1_7').output
    pooling1_8out = model8.get_layer('pooling1_8').output

    pooling2_1out = model1.get_layer('pooling2').output
    pooling2_2out = model2.get_layer('pooling2_2').output
    pooling2_3out = model3.get_layer('pooling2_3').output
    pooling2_4out = model4.get_layer('pooling2_4').output
    pooling2_5out = model5.get_layer('pooling2_5').output
    pooling2_6out = model6.get_layer('pooling2_6').output
    pooling2_7out = model7.get_layer('pooling2_7').output
    pooling2_8out = model8.get_layer('pooling2_8').output

    pooling3_1out = model1.get_layer('pooling3').output
    pooling3_2out = model2.get_layer('pooling3_2').output
    pooling3_3out = model3.get_layer('pooling3_3').output
    pooling3_4out = model4.get_layer('pooling3_4').output
    pooling3_5out = model5.get_layer('pooling3_5').output
    pooling3_6out = model6.get_layer('pooling3_6').output
    pooling3_7out = model7.get_layer('pooling3_7').output
    pooling3_8out = model8.get_layer('pooling3_8').output

    pooling4_1out = model1.get_layer('pooling4').output
    pooling4_2out = model2.get_layer('pooling4_2').output
    pooling4_3out = model3.get_layer('pooling4_3').output
    pooling4_4out = model4.get_layer('pooling4_4').output
    pooling4_5out = model5.get_layer('pooling4_5').output
    pooling4_6out = model6.get_layer('pooling4_6').output
    pooling4_7out = model7.get_layer('pooling4_7').output
    pooling4_8out = model8.get_layer('pooling4_8').output


    r_diff0 = Subtract()([conv1_1out, conv1_2out])
    r_diff0 = Lambda(lambda x: K.abs(x))(r_diff0)
    x_diff0 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(r_diff0)
    x_diff0 = BatchNormalization()(x_diff0)
    x_diff0 = ReLU()(x_diff0)


    r_diff1 = Subtract()([conv2_1out, conv2_2out])
    r_diff1 = Lambda(lambda x: abs(x))(r_diff1)
    r_diff1 = Concatenate()([r_diff1, x_diff0])
    x_diff1 = Convolution2D(96, (1, 1), padding='same', kernel_initializer='he_normal')(r_diff1)
    x_diff1 = BatchNormalization()(x_diff1)
    x_diff1 = ReLU()(x_diff1)
    x_diff1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_diff1)

    r_diff2 = Subtract()([pooling1_1out, pooling1_2out])
    r_diff2 = Lambda(lambda x: abs(x))(r_diff2)
    r_diff2 = Concatenate()([r_diff2, x_diff1])
    x_diff2 = Convolution2D(160, (1, 1), padding='same', kernel_initializer='he_normal')(r_diff2)
    x_diff2 = BatchNormalization()(x_diff2)
    x_diff2 = ReLU()(x_diff2)
    x_diff2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_diff2)

    r_diff3 = Subtract()([pooling2_1out, pooling2_2out])
    r_diff3 = Lambda(lambda x: abs(x))(r_diff3)
    r_diff3 = Concatenate()([r_diff3, x_diff2])
    x_diff3 = Convolution2D(288, (1, 1), padding='same', kernel_initializer='he_normal')(r_diff3)
    x_diff3 = BatchNormalization()(x_diff3)
    x_diff3 = ReLU()(x_diff3)
    x_diff3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_diff3)

    r_diff4 = Subtract()([pooling3_1out, pooling3_2out])
    r_diff4 = Lambda(lambda x: abs(x))(r_diff4)
    r_diff4 = Concatenate()([r_diff4, x_diff3])
    x_diff4 = Convolution2D(544, (1, 1), padding='same', kernel_initializer='he_normal')(r_diff4)
    x_diff4 = BatchNormalization()(x_diff4)
    x_diff4 = ReLU()(x_diff4)
    x_diff4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_diff4)

    r_diff5 = Subtract()([pooling4_1out, pooling4_2out])
    r_diff5 = Lambda(lambda x: abs(x))(r_diff5)
    r_diff5 = Concatenate()([r_diff5, x_diff4])
    x_diff5 = Convolution2D(1056, (3, 3), padding='same', kernel_initializer='he_normal')(r_diff5)
    x_diff5 = BatchNormalization()(x_diff5)
    x_diff5 = ReLU()(x_diff5)
    x_diff_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_diff5)

    r_diff_fc0 = Flatten()(x_diff_5)
    r_diff_fc1 = Dense(2048, activation='relu')(r_diff_fc0)
    r_diff_fc2 = Dense(1024, activation='relu')(r_diff_fc1)
    r_diff_fc3 = Dense(512, activation='relu')(r_diff_fc2)
    r_diff_fc4 = Dense(512, activation='relu')(r_diff_fc3)
    r_diff_fc5 = Dense(128, activation='relu')(r_diff_fc4)
    r_diff_fc6 = Dense(128, activation='relu')(r_diff_fc5)
    r_diff_fc7 = Dense(64, activation='relu')(r_diff_fc6)
    r_diff_fc8 = Dense(1, activation='sigmoid')(r_diff_fc7)


    r2_diff0 = Subtract()([conv1_3out, conv1_4out])
    r2_diff0 = Lambda(lambda x: K.abs(x))(r2_diff0)
    x2_diff0 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(r2_diff0)
    x2_diff0 = BatchNormalization()(x2_diff0)
    x2_diff0 = ReLU()(x2_diff0)  # 64×64


    r2_diff1 = Subtract()([conv2_3out, conv2_4out])
    r2_diff1 = Lambda(lambda x: abs(x))(r2_diff1)
    r2_diff1 = Concatenate()([r2_diff1, x2_diff0])
    x2_diff1 = Convolution2D(96, (1, 1), padding='same', kernel_initializer='he_normal')(r2_diff1)
    x2_diff1 = BatchNormalization()(x2_diff1)
    x2_diff1 = ReLU()(x2_diff1)  # 64×64
    x2_diff1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_diff1)  # 32×32

    r2_diff2 = Subtract()([pooling1_3out, pooling1_4out])
    r2_diff2 = Lambda(lambda x: abs(x))(r2_diff2)
    r2_diff2 = Concatenate()([r2_diff2, x2_diff1])
    x2_diff2 = Convolution2D(160, (1, 1), padding='same', kernel_initializer='he_normal')(r2_diff2)
    x2_diff2 = BatchNormalization()(x2_diff2)
    x2_diff2 = ReLU()(x2_diff2)  # 32×32
    x2_diff2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_diff2)  # 16×16

    r2_diff3 = Subtract()([pooling2_3out, pooling2_4out])
    r2_diff3 = Lambda(lambda x: abs(x))(r2_diff3)
    r2_diff3 = Concatenate()([r2_diff3, x2_diff2])
    x2_diff3 = Convolution2D(288, (1, 1), padding='same', kernel_initializer='he_normal')(r2_diff3)
    x2_diff3 = BatchNormalization()(x2_diff3)
    x2_diff3 = ReLU()(x2_diff3)
    x2_diff3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_diff3)  # 8×8

    r2_diff4 = Subtract()([pooling3_3out, pooling3_4out])
    r2_diff4 = Lambda(lambda x: abs(x))(r2_diff4)
    r2_diff4 = Concatenate()([r2_diff4, x2_diff3])
    x2_diff4 = Convolution2D(544, (1, 1), padding='same', kernel_initializer='he_normal')(r2_diff4)
    x2_diff4 = BatchNormalization()(x2_diff4)
    x2_diff4 = ReLU()(x2_diff4)
    x2_diff4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_diff4)  # 4×4

    r2_diff5 = Subtract()([pooling4_3out, pooling4_4out])
    r2_diff5 = Lambda(lambda x: abs(x))(r2_diff5)
    r2_diff5 = Concatenate()([r2_diff5, x2_diff4])
    x2_diff5 = Convolution2D(1056, (3, 3), padding='same', kernel_initializer='he_normal')(r2_diff5)
    x2_diff5 = BatchNormalization()(x2_diff5)
    x2_diff5 = ReLU()(x2_diff5)
    x2_diff_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2_diff5)

    r2_diff_fc0 = Flatten()(x2_diff_5)
    r2_diff_fc1 = Dense(2048, activation='relu')(r2_diff_fc0)
    r2_diff_fc2 = Dense(1024, activation='relu')(r2_diff_fc1)
    r2_diff_fc3 = Dense(512, activation='relu')(r2_diff_fc2)
    r2_diff_fc4 = Dense(512, activation='relu')(r2_diff_fc3)
    r2_diff_fc5 = Dense(128, activation='relu')(r2_diff_fc4)
    r2_diff_fc6 = Dense(128, activation='relu')(r2_diff_fc5)
    r2_diff_fc7 = Dense(64, activation='relu')(r2_diff_fc6)
    r2_diff_fc8 = Dense(1, activation='sigmoid')(r2_diff_fc7)


    r3_diff0 = Subtract()([conv1_5out, conv1_6out])
    r3_diff0 = Lambda(lambda x: K.abs(x))(r3_diff0)
    x3_diff0 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(r3_diff0)
    x3_diff0 = BatchNormalization()(x3_diff0)
    x3_diff0 = ReLU()(x3_diff0)  # 64×64


    r3_diff1 = Subtract()([conv2_5out, conv2_6out])
    r3_diff1 = Lambda(lambda x: abs(x))(r3_diff1)
    r3_diff1 = Concatenate()([r3_diff1, x3_diff0])
    x3_diff1 = Convolution2D(96, (1, 1), padding='same', kernel_initializer='he_normal')(r3_diff1)
    x3_diff1 = BatchNormalization()(x3_diff1)
    x3_diff1 = ReLU()(x3_diff1)  # 64×64
    x3_diff1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_diff1)  # 32×32

    r3_diff2 = Subtract()([pooling1_5out, pooling1_6out])
    r3_diff2 = Lambda(lambda x: abs(x))(r3_diff2)
    r3_diff2 = Concatenate()([r3_diff2, x3_diff1])
    x3_diff2 = Convolution2D(160, (1, 1), padding='same', kernel_initializer='he_normal')(r3_diff2)
    x3_diff2 = BatchNormalization()(x3_diff2)
    x3_diff2 = ReLU()(x3_diff2)  # 32×32
    x3_diff2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_diff2)  # 16×16

    r3_diff3 = Subtract()([pooling2_5out, pooling2_6out])
    r3_diff3 = Lambda(lambda x: abs(x))(r3_diff3)
    r3_diff3 = Concatenate()([r3_diff3, x3_diff2])
    x3_diff3 = Convolution2D(288, (1, 1), padding='same', kernel_initializer='he_normal')(r3_diff3)
    x3_diff3 = BatchNormalization()(x3_diff3)
    x3_diff3 = ReLU()(x3_diff3)
    x3_diff3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_diff3)  # 8×8

    r3_diff4 = Subtract()([pooling3_5out, pooling3_6out])
    r3_diff4 = Lambda(lambda x: abs(x))(r3_diff4)
    r3_diff4 = Concatenate()([r3_diff4, x3_diff3])
    x3_diff4 = Convolution2D(544, (1, 1), padding='same', kernel_initializer='he_normal')(r3_diff4)
    x3_diff4 = BatchNormalization()(x3_diff4)
    x3_diff4 = ReLU()(x3_diff4)
    x3_diff4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_diff4)  # 4×4

    r3_diff5 = Subtract()([pooling4_5out, pooling4_6out])
    r3_diff5 = Lambda(lambda x: abs(x))(r3_diff5)
    r3_diff5 = Concatenate()([r3_diff5, x3_diff4])
    x3_diff5 = Convolution2D(1056, (3, 3), padding='same', kernel_initializer='he_normal')(r3_diff5)
    x3_diff5 = BatchNormalization()(x3_diff5)
    x3_diff5 = ReLU()(x3_diff5)
    x3_diff_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3_diff5)

    r3_diff_fc0 = Flatten()(x3_diff_5)
    r3_diff_fc1 = Dense(2048, activation='relu')(r3_diff_fc0)
    r3_diff_fc2 = Dense(1024, activation='relu')(r3_diff_fc1)
    r3_diff_fc3 = Dense(512, activation='relu')(r3_diff_fc2)
    r3_diff_fc4 = Dense(512, activation='relu')(r3_diff_fc3)
    r3_diff_fc5 = Dense(128, activation='relu')(r3_diff_fc4)
    r3_diff_fc6 = Dense(128, activation='relu')(r3_diff_fc5)
    r3_diff_fc7 = Dense(64, activation='relu')(r3_diff_fc6)
    r3_diff_fc8 = Dense(1, activation='sigmoid')(r3_diff_fc7)


    r4_diff0 = Subtract()([conv1_7out, conv1_8out])
    r4_diff0 = Lambda(lambda x: K.abs(x))(r4_diff0)
    x4_diff0 = Convolution2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(r4_diff0)
    x4_diff0 = BatchNormalization()(x4_diff0)
    x4_diff0 = ReLU()(x4_diff0)  # 64×64


    r4_diff1 = Subtract()([conv2_7out, conv2_8out])
    r4_diff1 = Lambda(lambda x: abs(x))(r4_diff1)
    r4_diff1 = Concatenate()([r4_diff1, x4_diff0])
    x4_diff1 = Convolution2D(96, (1, 1), padding='same', kernel_initializer='he_normal')(r4_diff1)
    x4_diff1 = BatchNormalization()(x4_diff1)
    x4_diff1 = ReLU()(x4_diff1)  # 64×64
    x4_diff1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x4_diff1)  # 32×32

    r4_diff2 = Subtract()([pooling1_7out, pooling1_8out])
    r4_diff2 = Lambda(lambda x: abs(x))(r4_diff2)
    r4_diff2 = Concatenate()([r4_diff2, x4_diff1])
    x4_diff2 = Convolution2D(160, (1, 1), padding='same', kernel_initializer='he_normal')(r4_diff2)
    x4_diff2 = BatchNormalization()(x4_diff2)
    x4_diff2 = ReLU()(x4_diff2)  # 32×32
    x4_diff2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x4_diff2)  # 16×16

    r4_diff3 = Subtract()([pooling2_7out, pooling2_8out])
    r4_diff3 = Lambda(lambda x: abs(x))(r4_diff3)
    r4_diff3 = Concatenate()([r4_diff3, x4_diff2])
    x4_diff3 = Convolution2D(288, (1, 1), padding='same', kernel_initializer='he_normal')(r4_diff3)
    x4_diff3 = BatchNormalization()(x4_diff3)
    x4_diff3 = ReLU()(x4_diff3)
    x4_diff3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x4_diff3)  # 8×8

    r4_diff4 = Subtract()([pooling3_7out, pooling3_8out])
    r4_diff4 = Lambda(lambda x: abs(x))(r4_diff4)
    r4_diff4 = Concatenate()([r4_diff4, x4_diff3])
    x4_diff4 = Convolution2D(544, (1, 1), padding='same', kernel_initializer='he_normal')(r4_diff4)
    x4_diff4 = BatchNormalization()(x4_diff4)
    x4_diff4 = ReLU()(x4_diff4)
    x4_diff4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x4_diff4)  # 4×4

    r4_diff5 = Subtract()([pooling4_7out, pooling4_8out])
    r4_diff5 = Lambda(lambda x: abs(x))(r4_diff5)
    r4_diff5 = Concatenate()([r4_diff5, x4_diff4])
    x4_diff5 = Convolution2D(1056, (3, 3), padding='same', kernel_initializer='he_normal')(r4_diff5)
    x4_diff5 = BatchNormalization()(x4_diff5)
    x4_diff5 = ReLU()(x4_diff5)
    x4_diff_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x4_diff5)

    r4_diff_fc0 = Flatten()(x4_diff_5)
    r4_diff_fc1 = Dense(2048, activation='relu')(r4_diff_fc0)
    r4_diff_fc2 = Dense(1024, activation='relu')(r4_diff_fc1)
    r4_diff_fc3 = Dense(512, activation='relu')(r4_diff_fc2)
    r4_diff_fc4 = Dense(512, activation='relu')(r4_diff_fc3)
    r4_diff_fc5 = Dense(128, activation='relu')(r4_diff_fc4)
    r4_diff_fc6 = Dense(128, activation='relu')(r4_diff_fc5)
    r4_diff_fc7 = Dense(64, activation='relu')(r4_diff_fc6)
    r4_diff_fc8 = Dense(1, activation='sigmoid')(r4_diff_fc7)

    diff_fci = Concatenate()([x_diff5, x2_diff5, x3_diff5, x4_diff5])
    x = Convolution2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(diff_fci)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    res_fc0 = Flatten()(x)
    res_fc1 = Dense(2048, activation='relu')(res_fc0)
    res_fc2 = Dense(1024, activation='relu')(res_fc1)
    res_fc3 = Dense(512, activation='relu')(res_fc2)
    res_fc4 = Dense(512, activation='relu')(res_fc3)
    res_fc5 = Dense(128, activation='relu')(res_fc4)
    res_fc6 = Dense(128, activation='relu')(res_fc5)
    res_fc7 = Dense(64, activation='relu')(res_fc6)
    res_fc8 = Dense(1, activation='sigmoid')(res_fc7)


    class_models = Model(
        inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input,
                model8.input], outputs=[res_fc8, r_diff_fc8, r2_diff_fc8, r3_diff_fc8, r4_diff_fc8])
    return class_models


