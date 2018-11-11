from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Cropping2D
from keras.layers import concatenate
from keras.layers import add
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.inception_v3 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

_BN_DECAY = 0.9997
_BN_EPSILON = 1e-3



class NASNet():
    @staticmethod
    def NASNetLarge(input_shape=(331, 331, 3),classes=100,weights="trained_model/inception.hdf5"):
        global _BN_DECAY, _BN_EPSILON
        _BN_DECAY = 0.9997
        _BN_EPSILON = 1e-3
        dropout=0.5
        weight_decay=5e-5
        use_auxiliary_branch=False
        include_top=True
        input_tensor=None
        pooling=None

        return NASNet_basic(input_shape,
                  penultimate_filters=4032,
                  nb_blocks=6,
                  stem_filters=96,
                  skip_reduction_layer_input=True,
                  use_auxiliary_branch=use_auxiliary_branch,
                  filters_multiplier=2,
                  dropout=dropout,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  default_size=331)

def NASNet_basic(input_shape=None,
           penultimate_filters=4032,
           nb_blocks=6,
           stem_filters=96,
           skip_reduction_layer_input=True,
           use_auxiliary_branch=False,
           filters_multiplier=2,
           dropout=0.5,
           weight_decay=5e-5,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           default_size=None):
    img_input = Input(shape=input_shape)
    filters = penultimate_filters // 24
    channel_dim = -1
    x = Conv2D(stem_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                           name='stem_bn1')(x)
    p = None
    x, p = _reduction_A(x, p, filters // (filters_multiplier ** 2), weight_decay, id='stem_1')
    x, p = _reduction_A(x, p, filters // filters_multiplier, weight_decay, id='stem_2')
    for i in range(nb_blocks):
        x, p = _normal_A(x, p, filters, weight_decay, id='%d' % (i))
    x, p0 = _reduction_A(x, p, filters * filters_multiplier, weight_decay, id='reduce_%d' % (nb_blocks))

    
    if use_auxiliary_branch:
        auxiliary_x = _add_auxiliary_head(x, classes, weight_decay, pooling, include_top)

    p = p0 if not skip_reduction_layer_input else p

    for i in range(nb_blocks):
        x, p = _normal_A(x, p, filters * filters_multiplier ** 2, weight_decay, id='%d' % (2 * nb_blocks + i + 1))
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='predictions')(x)


    inputs = img_input
    # Create model.
    if use_auxiliary_branch:
        model = Model(inputs, [x, auxiliary_x], name='NASNet_with_auxiliary')
    else:
        model = Model(inputs, x, name='NASNet')

    return model


def _add_auxiliary_head(x, classes, weight_decay, pooling, include_top):
    '''Adds an auxiliary head for training the model
    From section A.7 "Training of ImageNet models" of the paper, all NASNet models are
    trained using an auxiliary classifier around 2/3 of the depth of the network, with
    a loss weight of 0.4
    # Arguments
        x: input tensor
        classes: number of output classes
        weight_decay: l2 regularization weight
    # Returns
        a keras Tensor
    '''
    img_height = 1 if K.image_data_format() == 'channels_last' else 2
    img_width = 2 if K.image_data_format() == 'channels_last' else 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('auxiliary_branch'):
        auxiliary_x = Activation('relu')(x)
        auxiliary_x = AveragePooling2D((5, 5), strides=(3, 3), padding='valid', name='aux_pool')(auxiliary_x)
        auxiliary_x = Conv2D(128, (1, 1), padding='same', use_bias=False, name='aux_conv_projection',
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(auxiliary_x)
        auxiliary_x = BatchNormalization(axis=channel_axis, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                        name='aux_bn_projection')(auxiliary_x)
        auxiliary_x = Activation('relu')(auxiliary_x)

        auxiliary_x = Conv2D(768, (auxiliary_x._keras_shape[img_height], auxiliary_x._keras_shape[img_width]),
                            padding='valid', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay), name='aux_conv_reduction')(auxiliary_x)
        auxiliary_x = BatchNormalization(axis=channel_axis, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                        name='aux_bn_reduction')(auxiliary_x)
        auxiliary_x = Activation('relu')(auxiliary_x)

        if include_top:
            auxiliary_x = Flatten()(auxiliary_x)
            auxiliary_x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay),
                                name='aux_predictions')(auxiliary_x)
        else:
            if pooling == 'avg':
                auxiliary_x = GlobalAveragePooling2D()(auxiliary_x)
            elif pooling == 'max':
                auxiliary_x = GlobalMaxPooling2D()(auxiliary_x)

    return auxiliary_x

def _reduction_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''
    """"""
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('reduction_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='reduction_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_1_%s' % id)
            x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left2_%s' % id)(h)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left3_%s' % id)(h)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right3_%s' % id)
            x3 = add([x3_1, x3_2], name='reduction_add3_%s' % id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % id)(x1)
            x4 = add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), weight_decay=weight_decay, id='reduction_left4_%s' % id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_right5_%s' % id)(h)
            x5 = add([x5_1, x5_2], name='reduction_add4_%s' % id)

        x = concatenate([x2, x3, x4, x5], axis=channel_dim, name='reduction_concat_%s' % id)
        return x, ip

def _normal_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('normal_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='normal_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                         id='normal_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, weight_decay=weight_decay, id='normal_right1_%s' % id)
            x1 = add([x1_1, x1_2], name='normal_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), weight_decay=weight_decay, id='normal_left2_%s' % id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), weight_decay=weight_decay, id='normal_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='normal_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % (id))(h)
            x3 = add([x3, p], name='normal_add_3_%s' % id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (id))(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_right4_%s' % (id))(p)
            x4 = add([x4_1, x4_2], name='normal_add_4_%s' % id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, weight_decay=weight_decay, id='normal_left5_%s' % id)
            x5 = add([x5, h], name='normal_add_5_%s' % id)

        x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s' % id)
    return x, ip

def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), weight_decay=5e-5, id=None):
    '''Adds 2 blocks of [relu-separable conv-batchnorm]
    # Arguments:
        ip: input tensor
        filters: number of output filters per layer
        kernel_size: kernel size of separable convolutions
        strides: strided convolution for downsampling
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('separable_conv_block_%s' % id):
        x = Activation('relu')(ip)
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_1_bn_%s" % (id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_2_bn_%s" % (id))(x)
    return x

def _adjust_block(p, ip, filters, weight_decay=5e-5, id=None):
    '''
    Adjusts the input `p` to match the shape of the `input`
    or situations where the output number of filters needs to
    be changed
    # Arguments:
        p: input tensor which needs to be modified
        ip: input tensor whose shape needs to be matched
        filters: number of output filters to be matched
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        an adjusted Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    img_dim = 2 if K.image_data_format() == 'channels_first' else -2

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p._keras_shape[img_dim] != ip._keras_shape[img_dim]:
            with K.name_scope('adjust_reduction_block_%s' % id):
                p = Activation('relu', name='adjust_relu_1_%s' % id)(p)

                p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_1_%s' % id)(p)
                p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_1_%s' % id, kernel_initializer='he_normal')(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_2_%s' % id)(p2)
                p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_2_%s' % id, kernel_initializer='he_normal')(p2)

                p = concatenate([p1, p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)

        elif p._keras_shape[channel_dim] != filters:
            with K.name_scope('adjust_projection_block_%s' % id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='adjust_conv_projection_%s' % id,
                           use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(p)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)
    return p