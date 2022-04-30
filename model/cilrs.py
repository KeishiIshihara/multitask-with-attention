import cv2
import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from skimage.io import imsave

try:
    from .base_model import Model
    from .common_layers import fc_block
    from .common_layers import driving_module_branched
    from .common_layers import upsample_light, upsample_heavy
    from .losses import mae, MSE
except:
    from base_model import Model
    from common_layers import fc_block
    from common_layers import driving_module_branched
    from common_layers import upsample_light, upsample_heavy
    from losses import mae, MSE


class CILRS(Model):
    def __init__(
        self,
        input_shape,
        name="CILRS",
        **kwargs
    ):
        super(CILRS, self).__init__(name=name, **kwargs)
        self.input_size = tuple(input_shape)

        self.branch_names = ["Follow", "Left", "Right", "Straight"]
        self.branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]
        self.num_branch = len(self.branch_names)
        self.num_output = len(self.branch_config[0])
        self.nav_cmd_shape = (self.num_branch,)

        self._has_built = False
        self.gradcam_model = None

    def call(self, input_images, input_nav_cmd, input_speed, training=None):
        outputs = self.model([input_images, input_nav_cmd, input_speed], training=training)
        return outputs

    def build_model(self, plot=False, **kwargs):
        self._has_built = True

        ResNet, _ = Classifiers.get('resnet34')
        resnet = ResNet(input_shape=self.input_size, weights=None, include_top=False)
        resnet = tf.keras.Model(inputs=resnet.inputs, outputs=resnet.outputs, name='resnet34')

        # perception
        inputs = tf.keras.layers.Input(shape=self.input_size, name='input_image')
        z = resnet(inputs)  # (None, 4, 7, 512)

        x = z
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        z_ave = tf.keras.layers.GlobalAveragePooling2D()(x)
        z_max = tf.keras.layers.GlobalMaxPooling2D()(x)
        z_flatten = z_ave + z_max

        # input speed
        input_speed = tf.keras.layers.Input(shape=(1,), name='input_speed')
        x = fc_block(input_speed, 64, dropout=0.3)
        speed_encoded = fc_block(x, 64, dropout=0.3)

        # latent layer
        j = tf.keras.layers.Concatenate(axis=-1)([z_flatten, speed_encoded])

        # speed branch
        x = fc_block(z_flatten, 128, dropout=0.3)
        x = fc_block(x, 128, dropout=0.3)
        speed = tf.keras.layers.Dense(1, activation='sigmoid', name='estimated_speed')(x)

        input_nav_cmd = tf.keras.layers.Input(self.nav_cmd_shape, name='input_nav_cmd')
        driving_module = driving_module_branched(input_shape=j.shape[1:], len_sequence=1)
        control_dict = driving_module([j, input_nav_cmd])
        self.model = tf.keras.Model(
            inputs=[inputs, input_nav_cmd, input_speed],
            outputs={
                'steer': control_dict['steer'],
                'throttle': control_dict['throttle'],
                'brake': control_dict['brake'],
                'speed': speed,
                'latent_features': z,
            },
            name='CILRS',
        )

        if plot:
            self.plot_model(self.model, 'model.png')

    def loss_fn(self, outputs, targets, loss_weights, class_weights):
        steer_loss, steer_losses = MSE(targets['steer'], outputs['steer'])
        throttle_loss, throttle_losses = MSE(targets['throttle'], outputs['throttle'])
        brake_loss, brake_losses = MSE(targets['brake'], outputs['brake'])
        steer_loss = class_weights['controls']['steer'] * steer_loss
        throttle_loss = class_weights['controls']['throttle'] * throttle_loss
        brake_loss = class_weights['controls']['brake'] * brake_loss
        # Control loss
        controls_loss = steer_loss + throttle_loss + brake_loss
        controls_loss = loss_weights['controls'] * controls_loss

        speed_loss = mae(targets['speed'], outputs['speed'])
        speed_loss = loss_weights['speed'] * speed_loss

        total_loss = controls_loss + speed_loss

        every_single_sample_losses = (class_weights['controls']['steer'] * steer_losses
                                      + class_weights['controls']['throttle'] * throttle_losses
                                      + class_weights['controls']['brake'] * brake_losses) * (1. / 3)

        return {
            'steer_loss': steer_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
            'control_loss': controls_loss,
            'speed_loss': speed_loss,
            'total_loss': total_loss,
            'all_sample_losses': every_single_sample_losses,
        }

    def metrics(self, outputs, targets):
        steer_mae = mae(outputs['steer'], targets['steer'][0])
        throttle_mae = mae(outputs['throttle'], targets['throttle'][0])
        brake_mae = mae(outputs['brake'], targets['brake'][0])
        controls_metrics = (steer_mae + throttle_mae + brake_mae) / 3
        speeds_metrics = mae(outputs['speed'], targets['speed'])

        total_metrics = (controls_metrics + speeds_metrics) / 2

        return {
            'steer_mae': steer_mae,
            'throttle_mae': throttle_mae,
            'brake_mae': brake_mae,
            'control_mae': controls_metrics,
            'speed_mae': speeds_metrics,
            'total_metrics': total_metrics,
        }

    def gradcam(self, x, filename='cilrs_{target}.png', indices=None, *args, **kwargs):
        """GradCAM on batch data
        Args:
            x: batch of inputs
            filename: png filename of the saliency maps to be saved. `target` argment is given to format method.
        Return:
            heatmaps: list of heatmap images
            aligned_heatmap: an image containing all heatmap images over input images
        """
        assert self._has_built, 'model has not built yet, call model.build_model() first.'

        print('Note: make sure you have loaded a certain checkpoints to the model')

        filename = filename.format(target='control')

        print('preparing gradcam-specific models..')

        self.gradcam_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs={
                'latent_features': self.model.output['latent_features'],
                'steer': self.model.output['steer'],
                'throttle': self.model.output['throttle'],
                'brake': self.model.output['brake'],
            },
            name='gradcam_model',
        )

        # perform gradcam on control
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.gradcam_model(list(x.values()))
            last_conv_layer_output = outputs['latent_features']
            tape.watch(last_conv_layer_output)

        pooled_grads = 0
        for target in ['steer', 'throttle', 'brake']:
            grads = tape.gradient(outputs[target], last_conv_layer_output)
            grads = tf.reduce_mean(grads, axis=(1, 2))
            pooled_grads += grads

        # (batch, h, w, c), (batch, c) -> (batch, h, w, c)
        last_conv_layer_output = tf.einsum('ijkl,il->ijkl', last_conv_layer_output, pooled_grads)
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # heatmaps over input images
        _heatmap = []
        for i, h in enumerate(heatmap):
            h = np.maximum(h, 0) / np.max(h)
            h = cv2.applyColorMap(np.uint8(cv2.resize(h, (384, 160)) * 255), cv2.COLORMAP_JET)
            h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
            h = x['input_images'][i] + h / 255
            h = np.maximum(h, 0) / np.max(h)
            h = np.uint8(h * 255)
            _heatmap.append(h)

        heatmaps = np.uint8(_heatmap)
        if indices is not None:
            heatmaps = heatmaps[indices]

        # align
        n_heatmaps = len(heatmaps)
        w_heatmaps = int(np.ceil(np.sqrt(n_heatmaps)))
        h_heatmaps = int(np.ceil(n_heatmaps / w_heatmaps))

        # black images for the remainder
        n_remainder = w_heatmaps * h_heatmaps - n_heatmaps
        if n_remainder > 0:
            black_image = np.zeros((160, 384, 3), dtype=np.uint8)
            _heatmaps = np.concatenate([heatmaps, [black_image] * n_remainder], axis=0)
        else:
            _heatmaps = heatmaps

        aligned_heatmap = np.uint8(
            np.vstack([np.hstack(_heatmaps[w_heatmaps * h : w_heatmaps * (h + 1)]) for h in range(h_heatmaps)]))

        if filename is not None:
            imsave(filename, aligned_heatmap)

        return heatmaps, aligned_heatmap
