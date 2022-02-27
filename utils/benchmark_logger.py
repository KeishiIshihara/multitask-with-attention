from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import logging
import numpy as np
import time
from datetime import datetime

from driving_benchmarks.version084.benchmark_tools.agent import Agent
from driving_benchmarks.version084.carla.client import VehicleControl


class LoggerAgent(Agent):
    def __init__(self, size='auto', has_display=0, save=False, save_path=None):
        self.has_display = has_display
        self.size = size
        self.save = save
        self.save_path = None
        if save:
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        self._start_time = 1e-2 # game time
        self._start_time_real = 1e-2
        self._timeout = float('inf')
        self.frame_rate = None
        self._episode_name = None
        self._wall = 0
        self._step = -1

        self.video_writer = None
        self._video_buffer = list()


    def init(self, timeout, episode_name, base_name, task_id, **kwargs):
        self.cleanup()

        if self.save:
            date = datetime.now().strftime('%Y%m%d_%H%M%S')
            (self.save_path / base_name).mkdir(parents=True, exist_ok=True)
            self.filename = str(self.save_path / base_name / f'{task_id}_{date}.mp4')
            logging.info(f'VIDEO FILENMAE: {self.filename}')
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # frame_rate = self.frame_rate * 10 if self.frame_rate is not None else 15.0
            self.frame_rate = 12.5
            if not self.size == 'auto':
                self.video_writer = cv2.VideoWriter(self.filename, self.fourcc, self.frame_rate, self.size, True)

        self._timeout = timeout
        self._episode_name = episode_name
        self._task_id = task_id
        self._step = -1
        self._video_buffer = list()

        self._start_time = kwargs.get('start_time') if kwargs.get('start_time') is not None \
            else time.perf_counter()
        self._start_time_real = time.perf_counter()


    def run_step(self, *args, **kwargs):
        pass

    def _process_model_output(self, steer, throttle, brake, directions):
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        if directions == 3:
            throttle *= 0.95

        elif directions == 4:
            steer *= 1.05
            throttle *= 0.95

        control = VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control

    def _update(self, image):
        # self._step += 1

        if self.save:
            assert image.ndim == 3
            self._video_buffer.append(image)
            if len(self._video_buffer) > 100: # Save once per 100 frames
                self._save_video()

    def _display(self, *args, **kwargs):
        pass

    def _save_video(self):
        if self.save and len(self._video_buffer):
            video_buffer = np.asarray(self._video_buffer)[...,::-1]
            if self.size == 'auto' and self.video_writer is None:
                self.size = (video_buffer.shape[2], video_buffer.shape[1])
                self.video_writer = cv2.VideoWriter(self.filename, self.fourcc, self.frame_rate, self.size, True)
            for img in video_buffer:
                self.video_writer.write(img.astype('uint8'))
            self._video_buffer = []

    def cleanup(self):
        self._save_video()
        if self.video_writer is not None:
            self.video_writer.release()

    @staticmethod
    def _write(text, i, j, canvas, fontsize=0.4, color='white', thickness=1, background=None):
        COLOR = {'white': (255,255,255), 'black': (0,0,0)}
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        if background is not None:
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontsize, thickness)
            text_w, text_h = text_size
            cv2.rectangle(canvas, (int(cols[j]), int(rows[i] - text_h)), (int(cols[j] + text_w), int(rows[i] + text_h*0.5)), COLOR[background], -1)
            cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, COLOR[color], thickness)
        else:
            cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, COLOR[color], thickness)

    @staticmethod
    def _stick_together(*args):
        h = min(*[x.shape[0] for x in args])
        imgs = []
        for x in args:
            r = h / x.shape[0]
            x = cv2.resize(x, (int(r * x.shape[1]), int(r * x.shape[0])))
            imgs.append(x)
        return np.concatenate(imgs, 1)

    @staticmethod
    def _show(display, title='debug'):
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        cv2.imshow(title, display)
        cv2.waitKey(1)

    @staticmethod
    def _contour(image, margin, color):
        image[:margin] = color
        image[-margin:] = color
        image[:, :margin] = color
        image[:, -margin:] = color