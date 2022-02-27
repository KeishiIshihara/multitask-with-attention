import tensorflow as tf
import tensorflow.keras.layers as nn


class RepeatLayer(tf.keras.layers.Layer):
    def __init__(self, num_repeats, name='repeat_layer'):
        super(RepeatLayer, self).__init__(name)
        self.num_repeats = num_repeats

    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, axis=-1), repeats=self.num_repeats, axis=-1)


def fc_block(inputs, units, dropout=None, dense_name=None):
    fc = tf.keras.layers.Dense(units, name=dense_name)(inputs)
    if dropout:
        fc = tf.keras.layers.Dropout(dropout)(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.BatchNormalization()(fc)
    return fc


def upsample_light(filters, size1, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(tf.keras.layers.Conv2D(filters, size1, strides=1, padding='same'))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(tf.keras.layers.UpSampling2D())
    if apply_dropout:
        result.add(tf.keras.layers.SpatialDropout2D(0.3))
    return result


def upsample_heavy(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.SpatialDropout2D(0.3))
    result.add(tf.keras.layers.ReLU())
    return result


def driving_module_branched(input_shape, len_sequence=1):
    branch_num = 4
    nav_cmd_shape = (4,)
    num_output = 3
    branch_names = ["Follow", "Left", "Right", "Straight"]
    # input
    input_features = tf.keras.layers.Input(input_shape, name='input_features')
    j = fc_block(input_features, 512, dropout=0.2)
    # navigation command input
    input_nav_cmd = tf.keras.layers.Input(nav_cmd_shape, name='input_nav_cmd')
    if len_sequence > 1:
        nav_cmd = RepeatLayer(len_sequence)(input_nav_cmd) # (batch, cmd, len_sequence)
    else:
        nav_cmd = input_nav_cmd # (bathch, cmd)
    branch_output = {
        'steer': [],
        'throttle': [],
        'brake': [],
    }
    for i in range(branch_num):
        k = fc_block(j, 256, dropout=0.3, dense_name='branch_{}'.format(branch_names[i]))
        # steer head
        x = fc_block(k, 128, dropout=0.2)
        x = fc_block(x, 82, dropout=0.2)
        steer = tf.keras.layers.Dense(len_sequence, activation='tanh')(x)
        # throttle head
        x = fc_block(k, 128, dropout=0.2)
        x = fc_block(x, 82, dropout=0.2)
        throttle = tf.keras.layers.Dense(len_sequence, activation='sigmoid')(x)
        # brake head
        x = fc_block(k, 128, dropout=0.2)
        x = fc_block(x, 82, dropout=0.2)
        brake = tf.keras.layers.Dense(len_sequence, activation='sigmoid')(x)

        steer = tf.keras.layers.Multiply()([steer, nav_cmd[:, i]])
        throttle = tf.keras.layers.Multiply()([throttle, nav_cmd[:, i]])
        brake = tf.keras.layers.Multiply()([brake, nav_cmd[:, i]])
        branch_output['steer'].append(steer)
        branch_output['throttle'].append(throttle)
        branch_output['brake'].append(brake)

    steer = tf.keras.layers.Add(name='steer')(branch_output['steer'])
    throttle = tf.keras.layers.Add(name='throttle')(branch_output['throttle'])
    brake = tf.keras.layers.Add(name='brake')(branch_output['brake'])
    driving_module = tf.keras.Model(
        inputs=[input_features, input_nav_cmd],
        outputs={
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
        },
        name='driving_module_branched'
    )
    return driving_module



#=============================================================
# Implementation of Convolutinal Block Attention Module(CBAM)
# Original paper: CBAM: Convolutional Block Attention Module
# https://arxiv.org/abs/1807.06521.
# code is taken from:
#=============================================================


class MLP(nn.Layer):
    """
    Multilayer perceptron block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Dense(
            units=mid_channels,
            input_dim=channels,
            name="fc1")
        self.activ = nn.ReLU()
        self.fc2 = nn.Dense(
            units=channels,
            input_dim=mid_channels,
            name="fc2")

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(nn.Layer):
    """
    CBAM channel gate block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 activation='sigmoid',
                 **kwargs):
        super(ChannelGate, self).__init__(**kwargs)
        self.avg_pool = nn.GlobalAvgPool2D(name="avg_pool")
        self.max_pool = nn.GlobalMaxPool2D(name="max_pool")
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio,
            name="mlp")
        if activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            self.activation = tf.nn.softmax
        self.reshape = tf.keras.layers.Reshape([1, 1, -1])

    def call(self, x, training=None):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.activation(att)
        att = self.reshape(att)
        x = x * att
        return x


class SpatialGate(nn.Layer):
    """
    CBAM spatial gate block.
    """
    def __init__(self, kernel_size=(7, 7), activation='sigmoid', **kwargs):
        super(SpatialGate, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size, strides=1, padding='same', name='conv')
        if activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            # self.activation = lambda x: tf.nn.softmax(x, axis=[1, 2])
            self.activation = lambda x: tf.keras.activations.softmax(x, axis=[1, 2])
        self.sp_att = None

    def call(self, x, training=None):
        att1 = tf.math.reduce_max(x, axis=-1, keepdims=True)
        att2 = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        att = tf.concat([att1, att2], axis=-1)
        att = self.conv(att, training=training)
        self.sp_att = self.activation(att)
        x = x * att
        return x


class CbamBlock(nn.Layer):
    """
    CBAM attention block for CBAM-ResNet.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 kernel_size=(7,7),
                 reduction_ratio=16,
                 activation='sigmoid',
                 **kwargs):
        super(CbamBlock, self).__init__(**kwargs)
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio,
            activation=activation,
            name="ch_gate")
        self.sp_gate = SpatialGate(
            kernel_size=kernel_size,
            activation=activation,
            name="sp_gate")

    def call(self, x, training=None):
        x = self.ch_gate(x, training=training)
        x = self.sp_gate(x, training=training)
        return x, self.sp_gate.sp_att
