import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Conv1D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, TensorBoard, History
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.layers import Conv3D
from functools import partial
import re
import math

import collections

from tensorflow.python.ops.gen_math_ops import imag

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size', 'include_top'])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

#swish = tf.keras.activations.swish 
def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(shape=[batch_size, 1, 1, 1, 1])
    binary_tensor = tf.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

def get_same_padding_conv3d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv3dDynamicSamePadding
    else:
        return partial(Conv3dStaticSamePadding, image_size=image_size)

class Conv3dDynamicSamePadding(Conv3D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.strides = [stride]
        self.strides = self.strides if len(self.strides) == 3 else [self.strides[0]] * 3
        super(Conv3dDynamicSamePadding, self).__init__(out_channels, kernel_size, strides=self.strides, dilation_rate=dilation, groups=groups, use_bias=bias)
        self.kernel_size = self.kernel_size if len(self.kernel_size) == 3 else [self.kernel_size[0]] * 3
        self.dilation = self.dilation_rate if len(self.dilation_rate) == 3 else [self.dilation_rate[0]] * 3
        self.paddings = self.kernel_size[0] // 2
    def __call__(self, inputs, training=None):
        ih, iw, iz = inputs.shape[1:-1]
        kh, kw, kz = self.kernel_size
        sh, sw, sz = self.strides
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.strides[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.strides[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.strides[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            #x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2])
            inputs = tf.pad(inputs, [[0, 0], [pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [pad_z // 2, pad_z - pad_z // 2], [0, 0]])
        return super(Conv3dDynamicSamePadding, self).call(inputs)
class Conv3dStaticSamePadding(Conv3D):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super(Conv3dStaticSamePadding, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.strides = self.strides if len(self.strides) == 3 else [self.strides[0]] * 3
        self.kernel_size = self.kernel_size if len(self.kernel_size) == 3 else [self.kernel_size[0]] * 3
        self.dilation = self.dilation_rate if len(self.dilation_rate) == 3 else [self.dilation_rate[0]] * 3
        self.paddings = self.kernel_size[0] // 2
        self.padding_value = 0
        self.image_size = image_size
        ih, iw, iz = image_size if type(image_size) == list else [image_size, image_size, image_size]
        kh, kw, kz = self.kernel_size
        sh, sw, sz = self.strides
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.strides[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.strides[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.strides[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            #self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2))
            self.static_padding = tf.keras.layers.ZeroPadding3D([(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_z // 2, pad_z - pad_z // 2)])
        else:
            # Identity()
            self.static_padding = tf.keras.layers.Lambda(lambda x: x)
    def call(self, inputs, training=None):
        x = self.static_padding(inputs)
        x = super(Conv3dStaticSamePadding, self).call(x)
        return x

def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 3 and options['s'][0] == options['s'][1] == options['s'][2]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d%d' % (block.strides[0], block.strides[1], block.strides[2]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet3d(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000, include_top=True):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s222_e1_i32_o16_se0.25', 'r2_k3_s222_e6_i16_o24_se0.25',
        'r2_k5_s222_e6_i24_o40_se0.25', 'r3_k3_s222_e6_i40_o80_se0.25',
        'r3_k5_s111_e6_i80_o112_se0.25', 'r4_k5_s222_e6_i112_o192_se0.25',
        'r1_k3_s111_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
        include_top=include_top,
    )

    return blocks_args, global_params

def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet3d(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params
