#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""

from keras.layers import *
from keras import initializers, regularizers, constraints
import math
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply
from keras import backend as K


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input_rgb.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def imp_eca(inputs_tensor=None, gamma=2, b=1, **kwargs):
    x_shape = inputs_tensor.shape.as_list()
    [batch_size, h, w, c] = x_shape

    h1 = int(abs((math.log(h, 2) + b) / gamma))
    k_h = h1 if h1 % 2 else h1 + 1

    w1 = int(abs((math.log(w, 2) + b) / gamma))
    k_w = w1 if w1 % 2 else w1 + 1

    c1 = int(abs((math.log(c, 2) + b) / gamma))
    k_c = c1 if c1 % 2 else c1 + 1

    h_tensor = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 3, 1]))(inputs_tensor)
    w_tensor = Lambda(lambda x: K.permute_dimensions(x, [0, 1, 3, 2]))(inputs_tensor)
    c_tensor = inputs_tensor

    h_tensor = GlobalAveragePooling2D()(h_tensor)
    w_tensor = GlobalAveragePooling2D()(w_tensor)
    c_tensor = GlobalAveragePooling2D()(c_tensor)

    h_tensor = Reshape((h, 1))(h_tensor)
    w_tensor = Reshape((w, 1))(w_tensor)
    c_tensor = Reshape((c, 1))(c_tensor)

    h_tensor = Conv1D(1, kernel_size=k_h, padding="same")(h_tensor)
    h_tensor = Activation('sigmoid')(h_tensor)
    h_tensor = Reshape((h, 1, 1))(h_tensor)
    w_tensor = Conv1D(1, kernel_size=k_w, padding="same")(w_tensor)
    w_tensor = Activation('sigmoid')(w_tensor)
    w_tensor = Reshape((1, w, 1))(w_tensor)
    c_tensor = Conv1D(1, kernel_size=k_c, padding="same")(c_tensor)
    c_tensor = Activation('sigmoid')(c_tensor)
    c_tensor = Reshape((1, 1, c))(c_tensor)
    output = multiply([inputs_tensor, h_tensor, w_tensor, c_tensor])
    return output



def d3_e_1g_mul2_reduce(x, gamma=2, b=1):
    bs, h, w, c = x.get_shape().as_list()

    h1 = int(abs((math.log(h, 2) + b) / gamma))
    k_h = h1 if h1 % 2 else h1 + 1

    w1 = int(abs((math.log(w, 2) + b) / gamma))
    k_w = w1 if w1 % 2 else w1 + 1

    c1 = int(abs((math.log(c, 2) + b) / gamma))
    k_c = c1 if c1 % 2 else c1 + 1


    input_c = x
    x_c1 = Lambda(lambda x: K.mean(x, axis=-1))(input_c)  ##H*W*1
    x_c2 = input_c ##H*W*C
    x_c1 = Reshape((h * w,1))(x_c1)
    x_c1 = Softmax(axis=1)(x_c1)  # [bs,h*w,1]
    x_c2 = Reshape((h * w,c))(x_c2)  # [bs,h*w,c]
    x_c1c2 = dot([x_c2, x_c1], axes=1) # [bs,c,1]
    x_c1c2 = Conv1D(1, kernel_size=k_c, padding="same")(x_c1c2)
    x_c1c2 = Activation('sigmoid')(x_c1c2)
    x_c1c2 = Reshape((1,1,c))(x_c1c2)


    input_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 3, 1]))(x)
    x_h1 = Lambda(lambda x: K.mean(x, axis=-1))(input_h)  ##W*C*1
    x_h2 = input_h  ##W*C*H
    x_h1 = Reshape((w * c, 1))(x_h1)
    x_h1 = Softmax(axis=1)(x_h1) # [bs,w*c,1]
    x_h2 = Reshape((w * c,h))(x_h2) # [bs,w*c,h]
    x_h1h2 = dot([x_h2, x_h1], axes=1) # [bs,h,1]
    x_h1h2 = Conv1D(1, kernel_size=k_h, padding="same")(x_h1h2)
    x_h1h2 = Activation('sigmoid')(x_h1h2)
    x_h1h2 = Reshape((h,1,1))(x_h1h2)

    input_w = Lambda(lambda x: K.permute_dimensions(x, [0, 1, 3, 2]))(x)
    x_w1 = Lambda(lambda x: K.mean(x, axis=-1))(input_w)  ##W*C*1
    x_w2 = input_w  ##H*C*W
    x_w1 = Reshape((h * c, 1))(x_w1)
    x_w1 = Softmax(axis=1)(x_w1)  # [bs,h*c,1]
    x_w2 = Reshape((h * c, w))(x_w2)  # [bs,h*c,w]
    x_w1w2 = dot([x_w2, x_w1], axes=1) # [bs,w,1]
    x_w1w2 = Conv1D(1, kernel_size=k_w, padding="same")(x_w1w2)
    x_w1w2 = Activation('sigmoid')(x_w1w2)
    x_w1w2 = Reshape((1,w,1))(x_w1w2)
    output = Multiply()([x, x_c1c2, x_h1h2, x_w1w2])
    return output




def large_margin_cosine_loss(y_true, y_pred, scale=20, margin=0.25):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def my_cosine_proximity(x):
    x1 = K.l2_normalize(x[0],axis=-1)
    x2 = K.l2_normalize(x[1],axis=-1)
    similar_x12 = K.sum(x1 * x2,axis=-1)
    fimally_similar = Lambda(lambda x: (x + 1)/2)(similar_x12)
    return fimally_similar
