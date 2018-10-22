from __future__ import print_function
from __future__ import absolute_import

import warnings
from functools import partial

from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
import os


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x

def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='Block35'`
        - Inception-ResNet-B: `block_type='Block17'`
        - Inception-ResNet-C: `block_type='Block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals before adding
            them to the shortcut branch.
        block_type: `'Block35'`, `'Block17'` or `'Block8'`, determines
            the network structure in the residual branch.
        block_idx: used for generating layer names.
        activation: name of the activation function to use at the end
            of the block (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'Block35'`,
            `'Block17'` or `'Block8'`.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 48, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 64, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name_fmt('ScaleSum'))([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x

class InceptionResNetV2():

    @staticmethod
    def inception_resnet_v2(input_shape,classes=100,weights="trained_model/inception_resnet_v2.hdf5"):

        img_input = Input(shape=input_shape)
            # Stem block: 35 x 35 x 192
        x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
        x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
        x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
        x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
        x = MaxPooling2D(3, strides=2, name='MaxPool_5a_3x3')(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 320
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        name_fmt = partial(_generate_layer_name, prefix='Mixed_5b')
        branch_0 = conv2d_bn(x, 96, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 48, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 64, 5, name=name_fmt('Conv2d_0b_5x5', 1))
        branch_2 = conv2d_bn(x, 64, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branch_pool = AveragePooling2D(3,
                                    strides=1,
                                    padding='same',
                                    name=name_fmt('AvgPool_0a_3x3', 3))(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, name=name_fmt('Conv2d_0b_1x1', 3))
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=channel_axis, name='Mixed_5b')(branches)

        # 10x Block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = _inception_resnet_block(x,
                                        scale=0.17,
                                        block_type='Block35',
                                        block_idx=block_idx)

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
        branch_0 = conv2d_bn(x,
                            384,
                            3,
                            strides=2,
                            padding='valid',
                            name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 256, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_1 = conv2d_bn(branch_1,
                            384,
                            3,
                            strides=2,
                            padding='valid',
                            name=name_fmt('Conv2d_1a_3x3', 1))
        branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

        # 20x Block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = _inception_resnet_block(x,
                                        scale=0.1,
                                        block_type='Block17',
                                        block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
        branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
        branch_0 = conv2d_bn(branch_0,
                            384,
                            3,
                            strides=2,
                            padding='valid',
                            name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1,
                            288,
                            3,
                            strides=2,
                            padding='valid',
                            name=name_fmt('Conv2d_1a_3x3', 1))
        branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 288, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2,
                            320,
                            3,
                            strides=2,
                            padding='valid',
                            name=name_fmt('Conv2d_1a_3x3', 2))
        branch_pool = MaxPooling2D(3,
                                strides=2,
                                padding='valid',
                                name=name_fmt('MaxPool_1a_3x3', 3))(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

        # 10x Block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = _inception_resnet_block(x,
                                        scale=0.2,
                                        block_type='Block8',
                                        block_idx=block_idx)
        x = _inception_resnet_block(x,
                                    scale=1.,
                                    activation=None,
                                    block_type='Block8',
                                    block_idx=10)

        # Final convolution block
        x = conv2d_bn(x, 1536, 1, name='Conv2d_7b_1x1')

        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - 0.8, name='Dropout')(x)
        x = Dense(classes, name='Logits')(x)
        x = Activation('softmax', name='Predictions')(x)

        # Create model.
        model = Model(img_input, x, name='inception_resnet_v2')

        if os.path.isfile(weights):
            model.load_weights(weights)
            print("Model loaded")
        else:
            print("No model is found")

        return model