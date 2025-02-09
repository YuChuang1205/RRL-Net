#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""


from keras import backend as K
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import Model

def large_margin_cosine_loss(y_true, y_pred, scale=20, margin=0.25):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)



def p_mse_loss_0001(y_true, y_pred):
    p_para = 1
    mse_para = 0.001

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out

def p_mse_loss_0005(y_true, y_pred):
    p_para = 1
    mse_para = 0.005

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out


def p_mse_loss_001(y_true, y_pred):
    p_para = 1
    mse_para = 0.01

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out



def p_mse_loss_005(y_true, y_pred):
    p_para = 1
    mse_para = 0.05

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out


def p_mse_loss_01(y_true, y_pred):
    p_para = 1
    mse_para = 0.1

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out

def p_mse_loss_01_no_train(y_true, y_pred):
    p_para = 1
    mse_para = 0.1

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    vgg_model.trainable = False
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out


def p_mse_loss_05(y_true, y_pred):
    p_para = 1
    mse_para = 0.5

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out


def p_mse_loss_1(y_true, y_pred):
    p_para = 1
    mse_para = 1

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out


def p_mse_loss_5(y_true, y_pred):
    p_para = 1
    mse_para = 5

    mse_loss = mse_para * K.mean(K.square(y_true - y_pred))

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[20].output)
    y_true_3 = K.concatenate([y_true, y_true, y_true], axis=-1)
    y_pred_3 = K.concatenate([y_pred, y_pred, y_pred], axis=-1)

    y_true_feature = vgg_model(y_true_3)
    y_pred_feature = vgg_model(y_pred_3)

    p_loss = p_para * K.mean(K.square(y_true_feature - y_pred_feature))

    loss_out = mse_loss + p_loss
    return loss_out

