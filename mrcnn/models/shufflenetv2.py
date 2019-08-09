import os
from keras import backend as K
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Input, Dense
from keras.layers import MaxPool2D, AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np
from mrcnn.models.utils import obtain_input_shape


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip],
                   name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1, train_bn=None, name=""):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = '{}/stage{}/block{}'.format(name, stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio * 0.5)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1,
               padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x, training=train_bn)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides,
                        padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x, training=train_bn)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1,
               padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x, training=train_bn)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(
            axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                             name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(
            axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2, training=train_bn)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1,
                    padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(
            axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2, training=train_bn)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(
            axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle,
                 name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, out_channel, bottleneck_ratio, repeat=1, stage=1, name=""):
    x = shuffle_unit(x, out_channels=out_channel,
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1, name=name)

    for i in range(repeat):
        x = shuffle_unit(x, out_channels=out_channel, strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(i + 2), name=name)

    return x


def ShuffleNetV2(include_top=True,
                 return_stages=False,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='avg',
                 input_shape=(224, 224, 3),
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=1,
                 classes=1000, train_bn=None):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(
        scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                     data_format=K.image_data_format())
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units),
                              dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    # calculate output channels for each stage
    out_channels_in_stage *= out_dim_stage_two[scale_factor]
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= bottleneck_ratio
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    stages = []
    prefix = 'shufflenetv2'
    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name=prefix + '/stage1/conv1')(img_input)
    stages.append(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                  padding='same', name=prefix + '/stage1/maxpool1')(x)
    stages.append(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage[stage + 1],
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2, name=prefix)
        stages.append(x)

    if scale_factor < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1,
               name=prefix + 'conv5', activation='relu')(x)
    stages[-1] = x

    if return_stages:
        return tuple(stages)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name=prefix + 'global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name=prefix + 'global_max_pool')(x)

    if include_top:
        x = Dense(classes, name=prefix + 'fc')(x)
        x = Activation('softmax', name=prefix + 'softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    return model


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ShuffleNetV2(include_top=False, return_stages=True,
                         input_shape=(224, 224, 3), scale_factor=2.0)

    model.summary()
