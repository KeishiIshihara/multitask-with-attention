import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import tensorflow as tf
import time

from .cilrs import CILRS
from .mt import DrivingModule
from .baseline import Baseline
from .mta import MTA
from utils.benchmark_logger import LoggerAgent
from utils.common import COMMAND_CONVERTER, COMMAND
from utils.common import TL, COLOR_cityscapes


class BaselineAgent(LoggerAgent):
    def __init__(
        self,
        input_shape,
        weight_path,
        len_sequence_output=1,
        has_display=0,
        save_video=False,
        save_path=None,
        debug_size=(1000, 800),
        **kwargs
    ):
        self.model = Baseline(input_shape, len_sequence_output=len_sequence_output)
        self.model.build_model()
        self.model.load_weights(weight_path=weight_path)
        self.speed_downscale_factor = 11
        self.debug_size = debug_size
        super(BaselineAgent, self).__init__(has_display=has_display, save=save_video, save_path=save_path, size='auto')

    def run_step(self, measurements, sensor_data, directions, target, env):
        self._step += 1
        self._wall = (measurements.game_timestamp - self._start_time) / 1000
        self._wall_real = time.perf_counter() - self._start_time_real
        self.frame_rate = self._step / self._wall_real
        image = sensor_data['CameraRGB'].data
        image = image.astype(np.float32) / 255
        input_speed_scaled = measurements.player_measurements.forward_speed / self.speed_downscale_factor
        inputs = {
            'rgb': np.uint8(image*255),
            'speed': input_speed_scaled,
            'command': COMMAND.get(directions, '???'),
        }
        outputs = self.model.predict(
            {'input_images': image[np.newaxis,...],
             'input_speed': np.asarray([input_speed_scaled]),
             'input_nav_cmd': tf.one_hot([COMMAND_CONVERTER[directions]], 4)},
            training=False
        )
        control = self._process_model_output(outputs['steer'][0][0], outputs['throttle'][0][0], outputs['brake'][0][0], directions)
        seg = tf.argmax(tf.nn.softmax(outputs['segmentation'], axis=-1), axis=-1).numpy()
        seg = COLOR_cityscapes[np.uint8(seg[0])]
        dep = np.squeeze(outputs['depth'] * 255, axis=0).astype('uint8')
        dep = np.repeat(dep, 3, axis=-1)
        speed_pred = np.squeeze(outputs['speed'][0])
        tl_state = np.squeeze(tf.argmax(tf.nn.softmax(outputs['tl_state']), axis=-1))
        refinements = {
            'dep': dep,
            'seg': seg,
            'steer': round(float(control.steer), 3),
            'throttle': round(float(control.throttle), 3),
            'brake': round(float(control.brake), 3),
            'speed_p': round(float(speed_pred), 3),
            'tl_state': TL[int(tl_state)],
        }
        if self._step%2 == 0:
            full = self._display(inputs, refinements, sensor_data, measurements, env)
            self._update(full)
        if 'CameraDebug' in sensor_data and self.has_display > 1:
            debug = np.uint8(sensor_data['CameraDebug'].data)
            self._show(debug, 'debug')
        return control

    def _display(self, inputs, refinements, sensor_data, measurements, env):
        if 'CameraDebug' in sensor_data:
            debug_canvas = sensor_data['CameraDebug'].data
        else:
            debug_canvas = None
        command = inputs['command']
        rgb = inputs['rgb']
        seg = refinements['seg']
        dep = refinements['dep']
        measured_speed_kmh = measurements.player_measurements.forward_speed * 3.6
        speed_pred_kmh = refinements['speed_p'] * self.speed_downscale_factor * 3.6
        steer = refinements['steer']
        throttle = refinements['throttle']
        brake = refinements['brake']
        tl_state = refinements['tl_state']

        if debug_canvas is None:
            fontsize_l = 0.8
            fontsize_s = 0.32

            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)

            full = self._stick_together(rgb, seg, dep)
            full = cv2.resize(full, (full.shape[1] * 5//2, full.shape[0] * 5//2))

        else:
            fontsize_s = 0.64
            h1, w1 = rgb.shape[:2]
            h1 = h1 * 6 // 5
            w1 = w1 * 6 // 5
            rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_NEAREST)
            dep = cv2.resize(dep, (w1, h1), interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, (w1, h1), interpolation=cv2.INTER_NEAREST)

            self._write(f'Network output', 1, 3, canvas=seg, fontsize=fontsize_s)
            self._write(f'Network output', 1, 3, canvas=dep, fontsize=fontsize_s, color='black')
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)

            debug_canvas = cv2.resize(debug_canvas, self.debug_size)
            h, w = debug_canvas.shape[:2]
            y_offset, x_offset = 20, 20
            y1, y2 = y_offset, y_offset + h1
            x1, x2 = x_offset, x_offset + w1
            debug_canvas.flags.writeable = True
            line_margin = 2
            line_color = (255, 255, 255)
            line_color_yellow = (240, 200, 0)
            self._contour(rgb, line_margin, line_color_yellow)
            self._contour(seg, line_margin, line_color)
            self._contour(dep, line_margin, line_color)
            debug_canvas[y1:y2, x1:x2] = rgb
            debug_canvas[-y2:-y1, x1:x2] = seg
            debug_canvas[-y2:-y1, -x2:-x1] = dep
            full = debug_canvas

        if self.has_display > 0:
            self._show(full, 'full')

        return full


class MTAAgent(LoggerAgent):
    def __init__(
        self,
        input_shape,
        weight_path,
        len_sequence_output=1,
        has_display=0,
        save_video=False,
        save_path=None,
        debug_size=(1000, 800),
        **kwargs
    ):
        self.model = MTA(input_shape, len_sequence_output=len_sequence_output)
        self.model.build_model()
        self.model.load_weights(weight_path=weight_path)
        self.speed_downscale_factor = 11
        self.debug_size = debug_size
        super(MTAAgent, self).__init__(has_display=has_display, save=save_video, save_path=save_path, size='auto')

    def run_step(self, measurements, sensor_data, directions, target, env):
        self._step += 1
        self._wall = (measurements.game_timestamp - self._start_time) / 1000
        self._wall_real = time.perf_counter() - self._start_time_real
        self.frame_rate = self._step / self._wall_real
        image = sensor_data['CameraRGB'].data
        image = image.astype(np.float32) / 255
        input_speed_scaled = measurements.player_measurements.forward_speed / self.speed_downscale_factor
        inputs = {
            'rgb': np.uint8(image*255),
            'speed': input_speed_scaled,
            'command': COMMAND.get(directions, '???'),
        }
        outputs = self.model.predict(
            {'input_images': image[np.newaxis,...],
             'input_speed': np.asarray([input_speed_scaled]),
             'input_nav_cmd': tf.one_hot([COMMAND_CONVERTER[directions]], 4)},
            training=False
        )
        control = self._process_model_output(outputs['steer'][0][0], outputs['throttle'][0][0], outputs['brake'][0][0], directions)
        seg = tf.argmax(tf.nn.softmax(outputs['segmentation'], axis=-1), axis=-1).numpy()
        seg = COLOR_cityscapes[np.uint8(seg[0])]
        dep = np.squeeze(outputs['depth'] * 255, axis=0).astype('uint8')
        dep = np.repeat(dep, 3, axis=-1)
        if outputs.get('speed') is not None:
            speed_pred = np.squeeze(outputs['speed'][0])
        else:
            speed_pred = None
        tl_state = np.squeeze(tf.argmax(tf.nn.softmax(outputs['tl_state']), axis=-1))
        if outputs.get('mask'):
            attention = outputs['mask'][0].numpy()
            attention = cv2.resize(attention*255, (attention.shape[1]*32, attention.shape[0]*32),
                                interpolation=cv2.INTER_NEAREST).astype('uint8')
            attention = np.tile(attention[... ,np.newaxis], (1, 1, 3))
        elif outputs.get('masks'):
            attention = outputs['masks']
        else:
            attention = None
        refinements = {
            'dep': dep,
            'seg': seg,
            'attention': attention,
            'steer': round(float(control.steer), 3),
            'throttle': round(float(control.throttle), 3),
            'brake': round(float(control.brake), 3),
            'speed_p': round(float(speed_pred), 3) if speed_pred is not None else None,
            'tl_state': TL[int(tl_state)],
        }
        if self._step%2 == 0:
            full = self._display(inputs, refinements, sensor_data, measurements, env)
            self._update(full)
        if 'CameraDebug' in sensor_data and self.has_display > 1:
            debug = np.uint8(sensor_data['CameraDebug'].data)
            self._show(debug, 'debug')
        return control

    def _display(self, inputs, refinements, sensor_data, measurements, env):
        if 'CameraDebug' in sensor_data:
            debug_canvas = sensor_data['CameraDebug'].data
        else:
            debug_canvas = None
        command = inputs['command']
        rgb = inputs['rgb']
        seg = refinements['seg']
        dep = refinements['dep']
        attention = refinements['attention']
        measured_speed_kmh = measurements.player_measurements.forward_speed * 3.6
        if refinements.get('speed_p') is not None:
            speed_pred_kmh = refinements['speed_p'] * self.speed_downscale_factor * 3.6
        else:
            speed_pred_kmh = None
        steer = refinements['steer']
        throttle = refinements['throttle']
        brake = refinements['brake']
        tl_state = refinements['tl_state']

        if debug_canvas is None:
            fontsize_l = 0.8
            fontsize_s = 0.32
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            if speed_pred_kmh is not None:
                self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)
            full = self._stick_together(rgb, seg, dep)
            full = cv2.resize(full, (full.shape[1] * 5//2, full.shape[0] * 5//2))
        else:
            fontsize_s = 0.64
            fontsize_l = 0.8
            h1, w1 = rgb.shape[:2]
            h2 = h1 * 6 // 5
            w2 = w1 * 6 // 5
            line_margin = 2
            line_color = (255, 255, 255)
            line_color_yellow = (240, 200, 0)
            rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_NEAREST)
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            if speed_pred_kmh is not None:
                self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)

            h = 160 * 5
            shape = debug_canvas.shape
            ratio = h / shape[0] * 0.95
            debug_canvas = cv2.resize(debug_canvas, (int(ratio * shape[1]), h))
            y_offset, x_offset = 20, 20
            y1, y2 = y_offset, y_offset + h2
            x1, x2 = x_offset, x_offset + w2
            self._contour(rgb, line_margin, line_color_yellow)
            debug_canvas.flags.writeable = True
            debug_canvas[y1:y2, x1:x2] = rgb
            def att_map_postprocess(np_array, size):
                att = cv2.resize(np_array, size, interpolation=cv2.INTER_NEAREST)
                att = cv2.applyColorMap(np.uint8(att * 255), cv2.COLORMAP_JET)
                return cv2.cvtColor(att, cv2.COLOR_BGR2RGB)
            att_maps_1 = []
            for task in ['semantic', 'depth']:
                _masks = []
                _att_maps = []
                for stage in ['stage_1', 'stage_2', 'stage_3', 'stage_4']:
                    mask = attention[task].get(stage)
                    att_map = att_map_postprocess(mask.numpy()[0], (w1, h1))
                    self._write(f'{task} : {stage}', 1, 2, canvas=att_map, color='white', fontsize=fontsize_l, thickness=1, background='black')
                    _att_maps.append(att_map)
                att_maps_1.append(np.vstack(_att_maps))
            att_maps_1 = np.hstack(att_maps_1)
            self._write(f'Network output', 1, 2, canvas=seg, fontsize=fontsize_l, thickness=1)
            self._write(f'Network output', 1, 2, canvas=dep, fontsize=fontsize_l, color='black', thickness=1)
            full_stack_right = np.vstack([np.hstack([seg, dep]), att_maps_1])
            # control att map
            att_map = attention['control']['stage_4'].numpy()[0]
            att_map = att_map_postprocess(att_map, (w2, h2))
            self._write(f'control : stage_4', 1, 2, canvas=att_map, color='white', background='black', fontsize=fontsize_l, thickness=1)
            debug_canvas[-y2:-y1, x1:x2] = att_map
            # tl att map
            att_map = attention['tl']['stage_4'].numpy()[0]
            att_map = att_map_postprocess(att_map, (w2, h2))
            self._write(f'traffic light : stage_4', 1, 2, canvas=att_map, color='white', background='black', fontsize=fontsize_l, thickness=1)
            debug_canvas[-y2:-y1, -x2:-x1] = att_map
            # full stack
            full = np.uint8(np.hstack([debug_canvas, full_stack_right]))
            line_margin = 1
            full[:, -line_margin:, :] = line_color # right
            full[:line_margin, -w1*2:] = line_color # right side top
            full[-line_margin:, -w1*2:] = line_color # right side bottom
            full[:, -w1-line_margin:-w1+line_margin, :] = line_color # ..seg | dep
            full[:, -w1*2-line_margin:-w1*2 +line_margin, :] = line_color # canvas | seg..
            full[ h1   - line_margin : h1   + line_margin, -w1*2:] = line_color # h 1
            full[ h1*2 - line_margin : h1*2 + line_margin, -w1*2:] = line_color # h 2
            full[ h1*3 - line_margin : h1*3 + line_margin, -w1*2:] = line_color # h 3
            full[ h1*4 - line_margin : h1*4 + line_margin, -w1*2:] = line_color # h 4
        if self.has_display > 0:
            self._show(full, 'full')
        return full


class CILRSAgent(LoggerAgent):
    def __init__(
        self,
        input_shape,
        weight_path,
        has_display=0,
        save_video=False,
        save_path=None,
        debug_size=(1000, 800),
        **kwargs
    ):
        self.model = CILRS(input_shape)
        self.model.build_model()
        self.model.load_weights(weight_path=weight_path)
        self.speed_downscale_factor = 11
        self.debug_size = debug_size
        super(CILRSAgent, self).__init__(has_display=has_display, save=save_video, save_path=save_path, size='auto')

    def run_step(self, measurements, sensor_data, directions, target, env):
        self._step += 1
        self._wall = (measurements.game_timestamp - self._start_time) / 1000
        self._wall_real = time.perf_counter() - self._start_time_real
        self.frame_rate = self._step / self._wall_real

        image = sensor_data['CameraRGB'].data
        input_speed_scaled = measurements.player_measurements.forward_speed / self.speed_downscale_factor
        inputs = {
            'rgb': np.uint8(image),
            'speed': input_speed_scaled,
            'command': COMMAND.get(directions, '???'),
        }
        image = np.float32(image) / 255
        outputs = self.model.predict(
            {'input_images': image[np.newaxis,...],
             'input_speed': np.asarray([input_speed_scaled]),
             'input_nav_cmd': tf.one_hot([COMMAND_CONVERTER[directions]], 4)},
            training=False
        )
        control = self._process_model_output(outputs['steer'][0][0], outputs['throttle'][0][0], outputs['brake'][0][0], directions)
        speed_pred = np.squeeze(outputs['speed'][0])
        refinements = {
            'steer': round(float(control.steer), 3),
            'throttle': round(float(control.throttle), 3),
            'brake': round(float(control.brake), 3),
            'speed_p': round(float(speed_pred), 3),
        }
        if self._step%2 == 0:
            full = self._display(inputs, refinements, sensor_data, measurements, env)
            self._update(full)
        if 'CameraDebug' in sensor_data and self.has_display > 1:
            debug = np.uint8(sensor_data['CameraDebug'].data)
            self._show(debug, 'debug')
        return control

    def _display(self, inputs, refinements, sensor_data, measurements, env):
        if 'CameraDebug' in sensor_data:
            debug_canvas = sensor_data['CameraDebug'].data
        else:
            debug_canvas = None
        command = inputs['command']
        rgb = inputs['rgb']
        measured_speed_kmh = measurements.player_measurements.forward_speed * 3.6
        speed_pred_kmh = refinements['speed_p'] * self.speed_downscale_factor * 3.6
        steer = refinements['steer']
        throttle = refinements['throttle']
        brake = refinements['brake']

        if debug_canvas is None:
            fontsize_l = 0.8
            fontsize_s = 0.32
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)
            full = rgb
            full = cv2.resize(full, (full.shape[1] * 5//2, full.shape[0] * 5//2))
        else:
            fontsize_s = 0.64
            h1, w1 = rgb.shape[:2]
            h1 = h1 * 6 // 5
            w1 = w1 * 6 // 5
            line_margin = 2
            line_color_yellow = (240, 200, 0)

            rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_NEAREST)
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            if speed_pred_kmh is not None:
                self._write(f'Speedp: {speed_pred_kmh:.1f}', 3, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)

            h = 160 * 5
            shape = debug_canvas.shape
            ratio = h / shape[0] * 0.95
            debug_canvas = cv2.resize(debug_canvas, (int(ratio * shape[1]), h))
            self._contour(rgb, line_margin, line_color_yellow)
            h, w = debug_canvas.shape[:2]
            y_offset, x_offset = 20, 20
            y1, y2 = y_offset, y_offset + h1
            x1, x2 = x_offset, x_offset + w1
            debug_canvas.flags.writeable = True
            debug_canvas[y1:y2, x1:x2] = rgb
            full = debug_canvas
        if self.has_display > 0:
            self._show(full, 'full')
        return full


class MTAgent(LoggerAgent):
    def __init__(
        self,
        input_shape,
        weight_path,
        has_display=0,
        save_video=False,
        save_path=None,
        debug_size=(1000, 800),
        **kwargs
    ):
        self.model = DrivingModule(input_shape)
        self.model.build_model(weight_file=None, plot=True)
        self.model.load_weights(weight_path)
        self.speed_downscale_factor = 11
        self.debug_size = debug_size
        super(MTAgent, self).__init__(has_display=has_display, save=save_video, save_path=save_path, size='auto')

    def run_step(self, measurements, sensor_data, directions, target, env):
        self._step += 1
        self._wall = (measurements.game_timestamp - self._start_time) / 1000
        self._wall_real = time.perf_counter() - self._start_time_real
        self.frame_rate = self._step / self._wall_real

        image = sensor_data['CameraRGB'].data
        input_speed_scaled = measurements.player_measurements.forward_speed / self.speed_downscale_factor
        inputs = {
            'rgb': np.uint8(image),
            'speed': input_speed_scaled,
            'command': COMMAND.get(directions, '???'),
        }
        image = np.float32(image) / 255
        outputs = self.model.predict(
            {'input_images': image[np.newaxis,...],
             'input_speed': np.asarray([input_speed_scaled]),
             'input_nav_cmd': tf.one_hot([COMMAND_CONVERTER[directions]], 4)},
            training=False
        )
        control = self._process_model_output(outputs['steer'][0][0], outputs['throttle'][0][0], outputs['brake'][0][0], directions)
        seg = tf.argmax(tf.nn.softmax(outputs['segmentation'], axis=-1), axis=-1).numpy()
        seg = COLOR_cityscapes[np.uint8(seg[0])]
        dep = np.squeeze(outputs['depth'] * 255, axis=0).astype('uint8')
        dep = np.repeat(dep, 3, axis=-1)
        tl_state = np.squeeze(tf.argmax(tf.nn.softmax(outputs['tl_state']), axis=-1))
        refinements = {
            'dep': dep,
            'seg': seg,
            'steer': round(float(control.steer), 3),
            'throttle': round(float(control.throttle), 3),
            'brake': round(float(control.brake), 3),
            'tl_state': TL[int(tl_state)],
        }
        if self._step%2 == 0:
            full = self._display(inputs, refinements, sensor_data, measurements, env)
            self._update(full)
        if 'CameraDebug' in sensor_data and self.has_display > 1:
            debug = np.uint8(sensor_data['CameraDebug'].data)
            self._show(debug, 'debug')
        return control

    def _display(self, inputs, refinements, sensor_data, measurements, env):
        if 'CameraDebug' in sensor_data:
            debug_canvas = sensor_data['CameraDebug'].data
        else:
            debug_canvas = None
        command = inputs['command']
        rgb = inputs['rgb']
        seg = refinements['seg']
        dep = refinements['dep']
        measured_speed_kmh = measurements.player_measurements.forward_speed * 3.6
        steer = refinements['steer']
        throttle = refinements['throttle']
        brake = refinements['brake']
        tl_state = refinements['tl_state']
        if debug_canvas is None:
            fontsize_l = 0.8
            fontsize_s = 0.32
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)
            full = self._stick_together(rgb, seg, dep)
            full = cv2.resize(full, (full.shape[1] * 5//2, full.shape[0] * 5//2))
        else:
            fontsize_s = 0.64
            h1, w1 = rgb.shape[:2]
            h1 = h1 * 6 // 5
            w1 = w1 * 6 // 5

            rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_NEAREST)
            dep = cv2.resize(dep, (w1, h1), interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, (w1, h1), interpolation=cv2.INTER_NEAREST)
            self._write(f'Network output', 1, 3, canvas=seg, fontsize=fontsize_s)
            self._write(f'Network output', 1, 3, canvas=dep, fontsize=fontsize_s, color='black')
            self._write(f'Command: {command}', 1, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Speedt: {measured_speed_kmh:.1f}', 2, 0, canvas=rgb,  fontsize=fontsize_s)
            self._write(f'Steer: {steer:.2f}', 5, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Throttle: {throttle:.2f}', 6, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Brake: {brake:.2f}', 7, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'TLp: {tl_state}', 9, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'R: {env["red"]}, G: {env["green"]}', 10, 0, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Timeout: {self._timeout:.1f}', 1, 6, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Time: {self._wall:.1f}', 2, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'FPS: {self.frame_rate:.1f}', 4, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.p: {env["col.p"]}', 5, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.v: {env["col.v"]}', 6, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Col.o: {env["col.oth"]}', 7, 7, canvas=rgb, fontsize=fontsize_s)
            self._write(f'Dist: {env["distance_to_goal"]:.1f}', 10, 7, canvas=rgb, fontsize=fontsize_s)

            debug_canvas = cv2.resize(debug_canvas, self.debug_size)
            h, w = debug_canvas.shape[:2]
            y_offset, x_offset = 20, 20
            y1, y2 = y_offset, y_offset + h1
            x1, x2 = x_offset, x_offset + w1
            debug_canvas.flags.writeable = True
            line_margin = 2
            line_color = (255, 255, 255)
            line_color_yellow = (240, 200, 0)
            self._contour(rgb, line_margin, line_color_yellow)
            self._contour(seg, line_margin, line_color)
            self._contour(dep, line_margin, line_color)
            debug_canvas[y1:y2, x1:x2] = rgb
            debug_canvas[-y2:-y1, x1:x2] = seg
            debug_canvas[-y2:-y1, -x2:-x1] = dep
            full = debug_canvas
        if self.has_display > 0:
            self._show(full, 'full')
        return full
