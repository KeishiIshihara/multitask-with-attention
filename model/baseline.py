import cv2
import numpy as np
import tensorflow as tf
from skimage.io import imsave

try:
    from .base_model import Model
    from .losses import mse, mae
    from .losses import weighted_sequence_mse, weighted_softmax_crossentropy
    from .common_layers import fc_block
    from .common_layers import driving_module_branched
    from .common_layers import upsample_light, upsample_heavy
    from .resnet import ModelsFactory
except:
    from base_model import Model
    from losses import mse, mae
    from losses import weighted_sequence_mse, weighted_softmax_crossentropy
    from common_layers import fc_block
    from common_layers import driving_module_branched
    from common_layers import upsample_light, upsample_heavy
    from resnet import ModelsFactory


class Baseline(Model):
    def __init__(
        self,
        input_shape,
        len_sequence_output,
        name="baseline",
        *args,
        **kwargs
    ):
        super(Baseline, self).__init__(name=name, *args, **kwargs)
        self.input_size = tuple(input_shape)
        self.len_sequence_output = len_sequence_output

        self.branch_names = ["Follow", "Left", "Right", "Straight"]
        self.branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]
        self.num_branch = len(self.branch_names)
        self.num_output = len(self.branch_config[0])
        self.nav_cmd_shape = (self.num_branch,)

        self._modules = dict()
        self._has_built = False
        self.gradcam_model = None

    def call(self, input_images, input_nav_cmd, input_speed, training=None):
        outputs = self.model([input_images, input_nav_cmd, input_speed], training=training)
        return outputs

    @tf.function
    def predict(self, inputs, training=False):
        outputs = self(**inputs, training=training)
        return outputs

    def build_model(self, plot=False, **kwargs):
        self._has_built = True

        # --------------------------------
        # *********  Modules *********
        # --------------------------------

        # ResNet
        ResNet = ModelsFactory.get('resnet34')
        resnet = ResNet(input_shape=self.input_size, weights=None, include_top=False)
        # Encoder
        self.Encoder = tf.keras.Model(inputs=resnet.inputs, outputs=resnet.outputs , name='encoder')
        self._modules['Encoder'] = self.Encoder
        latent_shape = self.Encoder.output.shape.as_list()[1:]
        latent_inputs = tf.keras.layers.Input(shape=latent_shape, name='latent_inputs')
        # SegNet
        up_stack = [
            upsample_light(256, 3, apply_dropout=True),
            upsample_heavy(128, 4, apply_dropout=True),
            upsample_heavy(96, 4, apply_dropout=True),
            upsample_heavy(64, 4),
        ]
        x = latent_inputs
        for up in up_stack:
            x = up(x)
        x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # added
        x = tf.keras.layers.ReLU()(x)
        segmentation = tf.keras.layers.Conv2D(13, 1, strides=1, padding='same')(x)  # This output should be unscaled
        self.SegNet = tf.keras.Model(inputs=latent_inputs, outputs=segmentation, name='segnet')
        self._modules['SegNet'] = self.SegNet

        # DepNet
        up_stack = [
            upsample_light(256, 3, apply_dropout=True),
            upsample_light(128, 3, apply_dropout=True),
            upsample_light(96, 3, apply_dropout=True),
            upsample_light(64, 3),
        ]
        x = latent_inputs
        for up in up_stack:
            x = up(x)
        x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # added
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')(x)  # Should I scale this output?
        x = tf.keras.layers.BatchNormalization()(x)  # added
        depth = tf.keras.layers.Activation('sigmoid')(x)
        self.DepthNet = tf.keras.Model(inputs=latent_inputs, outputs=depth, name='depthnet')
        self._modules['DepthNet'] = self.DepthNet

        # TrafficLight Classifier
        x = latent_inputs
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = fc_block(x, 128, 0.3)
        x = fc_block(x, 128, 0.2)
        tl_state = tf.keras.layers.Dense(4, name='tl_state_predicted')(x)  # Unscaled
        self.LightClassifier = tf.keras.Model(inputs=latent_inputs, outputs=tl_state, name='tl_classifier')
        self._modules['LightClassifier'] = self.LightClassifier

        # Speed Estimator
        x = latent_inputs
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = fc_block(x, 128, dropout=0.3)
        x = fc_block(x, 128, dropout=0.3)
        speed = tf.keras.layers.Dense(1, activation='sigmoid', name='estimated_speed')(x)
        self.SpeedEstimator = tf.keras.Model(inputs=latent_inputs, outputs=speed, name='speed_extimator')
        self._modules['SpeedEstimator'] = self.SpeedEstimator

        # Latent Feature Flatten Layers
        x = latent_inputs
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        z_1 = tf.keras.layers.GlobalMaxPooling2D()(x)
        z_2 = tf.keras.layers.GlobalAveragePooling2D()(x)
        flattened_feature = z_1 + z_2
        self.FlattenModule = tf.keras.Model(inputs=latent_inputs, outputs=flattened_feature, name='flatten_module')
        self._modules['FlattenModule'] = self.FlattenModule

        # Speed Encoder
        input_speed = tf.keras.layers.Input(shape=(1,), name='input_speed')
        x = fc_block(input_speed, 64, dropout=0.3)
        speed_encoded = fc_block(x, 64, dropout=0.3)
        self.SpeedEncoder = tf.keras.Model(inputs=input_speed, outputs=speed_encoded, name='speed_encoder')
        self._modules['SpeedEncoder'] = self.SpeedEncoder

        # ----------------------------------------
        # ****** Building Entire Network *******
        # ----------------------------------------

        inputs = tf.keras.layers.Input(shape=self.input_size, name='input_image')
        z = self.Encoder(inputs)

        # Decoders head
        segmentation = self.SegNet(z)
        depth = self.DepthNet(z)

        # TL classification
        tl_state = self.LightClassifier(z)

        # Speed Estimation head w/o speed features
        speed = self.SpeedEstimator(z)

        # Flatten latent feature
        z_flatten = self.FlattenModule(z)

        # Speed input
        speed_encoded = self.SpeedEncoder(input_speed)

        # Concate latent layer with speed features
        self.FusionLayer = tf.keras.layers.Concatenate(axis=-1, name='latent_fusion_layer')
        self._modules['FusionLayer'] = self.FusionLayer
        j = self.FusionLayer([z_flatten, speed_encoded])

        # Command input
        input_nav_cmd = tf.keras.layers.Input(self.nav_cmd_shape, name='input_nav_cmd')
        # Driving module head
        self.DrivingModule = driving_module_branched(input_shape=j.shape[1:], len_sequence=self.len_sequence_output)
        self._modules['DrivingModule'] = self.DrivingModule
        control_dict = self.DrivingModule([j, input_nav_cmd])

        self.model = tf.keras.Model(
            inputs=[inputs, input_nav_cmd, input_speed],
            outputs={
                'steer': control_dict['steer'],
                'throttle': control_dict['throttle'],
                'brake': control_dict['brake'],
                'speed': speed,
                'tl_state': tl_state,
                'segmentation': segmentation,
                'depth': depth,
                'latent_features': z,
            },
            name='baseline',
        )
        if plot:
            self.plot_model(self.model, 'model.png')

    def loss_fn(self, outputs, targets, loss_weights, class_weights):
        # Each control loss
        control_weights = class_weights['sequence_weight']
        steer_loss, steer_losses = weighted_sequence_mse(targets['steer'], outputs['steer'], control_weights)
        throttle_loss, throttle_losses = weighted_sequence_mse(
            targets['throttle'], outputs['throttle'], control_weights)
        brake_loss, brake_losses = weighted_sequence_mse(targets['brake'], outputs['brake'], control_weights)
        steer_loss = class_weights['controls']['steer'] * steer_loss
        throttle_loss = class_weights['controls']['throttle'] * throttle_loss
        brake_loss = class_weights['controls']['brake'] * brake_loss
        # Control loss
        controls_loss = steer_loss + throttle_loss + brake_loss
        controls_loss = loss_weights['controls'] * controls_loss
        # Speed estimation loss
        speed_loss = mae(y_true=targets['speed'], y_pred=outputs['speed'])
        speed_loss = loss_weights['speed'] * speed_loss
        # TL loss
        tl_loss = weighted_softmax_crossentropy(targets['tl_state'], outputs['tl_state'], class_weights['tl'])
        tl_loss = loss_weights['tl'] * tl_loss
        # Seg and depth loss
        seg_loss = weighted_softmax_crossentropy(tf.one_hot(
            targets['segmentation'], 13), outputs['segmentation'], class_weights['segmentation'])
        seg_loss = loss_weights['segmentation'] * seg_loss
        dep_loss = mse(targets['depth'], outputs['depth'])
        dep_loss = loss_weights['depth'] * dep_loss
        # Total loss
        total_loss = controls_loss + speed_loss + tl_loss + seg_loss + dep_loss
        # Each sample loss
        every_single_sample_losses = (class_weights['controls']['steer'] * steer_losses
                                      + class_weights['controls']['throttle'] * throttle_losses
                                      + class_weights['controls']['brake'] * brake_losses) * (1. / 3)

        return {
            'steer_loss': steer_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
            'control_loss': controls_loss,
            'speed_loss': speed_loss,
            'tl_loss': tl_loss,
            'seg_loss': seg_loss,
            'dep_loss': dep_loss,
            'total_loss': total_loss,
            'all_sample_losses': every_single_sample_losses,
        }

    def metrics(self, outputs, targets):
        # Control
        steer_mae = mae(targets['steer'], outputs['steer'])
        throttle_mae = mae(targets['throttle'], outputs['throttle'])
        brake_mae = mae(targets['brake'], outputs['brake'])
        controls_metrics = (steer_mae + throttle_mae + brake_mae) / 3
        # Speed
        speed_mae = mae(targets['speed'], outputs['speed'])
        # Seg and Depth
        equality = tf.equal(tf.cast(targets['segmentation'], tf.int64), tf.argmax(outputs['segmentation'], axis=-1))
        seg_accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        depth_mae = mae(targets['depth'], outputs['depth'])
        # TL
        tl_equality = tf.equal(tf.argmax(tf.cast(targets['tl_state'], tf.int64),
                               axis=-1), tf.argmax(outputs['tl_state'], axis=-1))
        tl_accuracy = tf.reduce_mean(tf.cast(tl_equality, tf.float32))
        # TOTAL
        total_metrics = (controls_metrics + speed_mae - tl_accuracy - seg_accuracy + depth_mae) / 5

        return {
            'steer_mae': steer_mae,
            'throttle_mae': throttle_mae,
            'brake_mae': brake_mae,
            'control_mae': controls_metrics,
            'speed_mae': speed_mae,
            'tl_acc': tl_accuracy,
            'seg_acc': seg_accuracy,
            'depth_mae': depth_mae,
            'total_metrics': total_metrics,
        }

    def gradcam(self, x, target_output='control', filename='mt_{target}.png', indices=None, *args, **kwargs):
        """GradCAM on batch data
        Args:
            x: batch of inputs
            target_layer: name of target layer to vizualize gradcam
            target_output: name of target output which grad will be calculated based on
            filename: png filename of the saliency maps to be saved. `target` argment is given to format method.
        Return:
            heatmaps: list of heatmap images
            aligned_heatmap: an image containing all heatmap images over input images
        """
        assert self._has_built, 'model has not built yet, call model.build_model() first.'
        assert target_output in [
            'tl_state', 'control'], f'target_output must be either `tl_state` or `control`, not {target_output}'

        print('Note: make sure you have loaded a certain checkpoints to the model')

        filename = filename.format(target=target_output)

        print('preparing gradcam-specific models..')

        self.gradcam_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs={
                'latent_features': self.model.output['latent_features'],
                'steer': self.model.output['steer'],
                'throttle': self.model.output['throttle'],
                'brake': self.model.output['brake'],
                'tl_state': self.model.output['tl_state'],
            },
            name='gradcam_model',
        )

        if target_output == 'tl_state':
            # perform gradcam on tl_state
            with tf.GradientTape() as tape:
                outputs = self.gradcam_model(list(x.values()))
                target_conv_layer_output = outputs['latent_features']
                tape.watch(target_conv_layer_output)
                preds = outputs[target_output]
                top_pred_index = tf.argmax(preds, axis=-1)
                top_class_channel = tf.gather_nd(preds, np.dstack([range(preds.shape[0]), top_pred_index])[0])

            grads = tape.gradient(top_class_channel, target_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
            # (batch, h, w, c), (batch, c) -> (batch, h, w, c)
            target_conv_layer_output = tf.einsum('ijkl,il->ijkl', target_conv_layer_output, pooled_grads)
            heatmap = np.mean(target_conv_layer_output, axis=-1)

        else:
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
