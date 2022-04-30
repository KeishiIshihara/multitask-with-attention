import argparse
import os
import pathlib

import cv2
import h5py
import numpy as np
import tensorflow as tf
from skimage.io import imsave

from model.baseline import Baseline
from model.cilrs import CILRS
from model.mt import DrivingModule
from model.mta import MTA


def load_scene_from_h5file(h5_dataset, num_frames, seed=None):
    if isinstance(seed, int):
        np.random.seed(seed)

    images, measure = [], []
    with h5py.File(h5_dataset, 'r') as h5file:
        episode_groups = list(h5file.keys())
        for _ in range(num_frames):
            eps = np.random.choice(episode_groups)
            label = h5file[str(pathlib.Path(eps) / 'measure')][:]
            episode_size = len(label)
            idx = np.random.randint(0, episode_size)
            images.append(h5file[str(pathlib.Path(eps) / 'rgb')][idx])
            measure.append(h5file[str(pathlib.Path(eps) / 'measure')][idx])

    images = np.float32(images) / 255
    measure = np.float32(measure)
    steer = measure[:, :1]
    throttle = measure[:, 1:2]
    brake = measure[:, 2:3]
    speed = measure[:, 3:4]
    nav_cmd = tf.one_hot(tf.cast(measure[:, 4], tf.uint8), 4)
    tl_state = tf.one_hot(tf.cast(measure[:, 5], tf.uint8), 4)

    return {
        'inputs': {
            'input_images': images,
            'input_nav_cmd': nav_cmd,
            'input_speed': speed,
        },
        'targets': {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'tl_state': tl_state,
        }
    }


def gradcam_MTA(
    data_dir,
    num_frames,
    seed,
    output_dir,
    width,
    height,
    indices,
):
    model = MTA((160, 384, 3), 1)
    model.build_model()
    model.load_weights(weight_path='ckpts/MTA/ckpt')

    dst_root = pathlib.Path(output_dir) / 'mta'
    dst_root.mkdir(exist_ok=True)

    data_dict = load_scene_from_h5file(data_dir, num_frames, seed)

    # ===================================
    # GradCam on control prediction
    # ===================================

    save_dir = dst_root / 'control'
    save_dir.mkdir(exist_ok=True)

    # gradcam on control prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        target_layer='z_control',
        target_output='control',
        filename=str(save_dir / 'mta_{target}.png'),
        indices=indices['control'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)

    # ===================================
    # GradCam on tl state classification
    # ===================================

    save_dir = dst_root / 'tl'
    save_dir.mkdir(exist_ok=True)

    # gradcam on tl state prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        target_layer='z_tl',
        target_output='tl_state',
        filename=str(save_dir / 'mta_{target}.png'),
        indices=indices['tl'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)


def gradcam_MT(
    data_dir,
    num_frames,
    seed,
    output_dir,
    width,
    height,
    indices,
):
    model = DrivingModule((160, 384, 3))
    model.build_model()
    model.load_weights(weight_path='ckpts/MT/ckpt')

    dst_root = pathlib.Path(output_dir) / 'mt'
    dst_root.mkdir(exist_ok=True)

    data_dict = load_scene_from_h5file(data_dir, num_frames, seed)

    # ===================================
    # GradCam on control prediction
    # ===================================

    save_dir = dst_root / 'control'
    save_dir.mkdir(exist_ok=True)

    # gradcam on control prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        target_layer='latent_features',
        target_output='control',
        filename=str(save_dir / 'mta_{target}.png'),
        indices=indices['control'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)

    # ===================================
    # GradCam on tl state classification
    # ===================================

    save_dir = dst_root / 'tl'
    save_dir.mkdir(exist_ok=True)

    # gradcam on tl state prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        target_layer='latent_features',
        target_output='tl_state',
        filename=str(save_dir / 'mt_{target}.png'),
        indices=indices['tl'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)


def gradcam_CILRS(
    data_dir,
    num_frames,
    seed,
    output_dir,
    width,
    height,
    indices,
):
    model = CILRS((160, 384, 3))
    model.build_model()
    model.load_weights(weight_path='ckpts/CILRS/ckpt')

    dst_root = pathlib.Path(output_dir) / 'cilrs'
    dst_root.mkdir(exist_ok=True)

    data_dict = load_scene_from_h5file(data_dir, num_frames, seed)

    # ===================================
    # GradCam on control prediction
    # ===================================

    save_dir = dst_root / 'control'
    save_dir.mkdir(exist_ok=True)

    # gradcam on control prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        filename=str(save_dir / 'cilrs_{target}.png'),
        indices=indices['control'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)


def gradcam_Baseline(
    data_dir,
    num_frames,
    seed,
    output_dir,
    width,
    height,
    indices,
):
    model = Baseline((160, 384, 3), 1)
    model.build_model()
    model.load_weights(weight_path='ckpts/baseline/ckpt')

    dst_root = pathlib.Path(output_dir) / 'baseline'
    dst_root.mkdir(exist_ok=True)

    data_dict = load_scene_from_h5file(data_dir, num_frames, seed)

    # ===================================
    # GradCam on control prediction
    # ===================================

    save_dir = dst_root / 'control'
    save_dir.mkdir(exist_ok=True)

    # gradcam on control prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        filename=str(save_dir / 'baseline_{target}.png'),
        indices=indices['control'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)

    # ===================================
    # GradCam on tl state classification
    # ===================================

    save_dir = dst_root / 'tl'
    save_dir.mkdir(exist_ok=True)

    # gradcam on tl state prediction
    heatmaps, _ = model.gradcam(
        data_dict['inputs'],
        filename=str(save_dir / 'baseline_{target}.png'),
        indices=indices['tl'],
    )

    # save every image and heatmap
    for i in range(len(heatmaps)):
        # save RGB
        image = data_dict['inputs']['input_images'][i] * 255
        image = cv2.resize(image, (width, height))
        imsave(f'{str(save_dir)}/{i}_CameraRGB.png', np.uint8(image))
        # save heatmap
        heatmap = heatmaps[i]
        heatmap = cv2.resize(heatmap, (width, height))
        imsave(f'{str(save_dir)}/{i}_heatmap.png', heatmap)


def main(args):
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    gradcam_MTA(
        data_dir=args.h5_dataset_path,
        num_frames=args.num_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        width=192,
        height=80,
        indices=args.indices,
    )

    gradcam_MT(
        data_dir=args.h5_dataset_path,
        num_frames=args.num_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        width=192,
        height=80,
        indices=args.indices,
    )

    gradcam_CILRS(
        data_dir=args.h5_dataset_path,
        num_frames=args.num_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        width=192,
        height=80,
        indices=args.indices,
    )

    gradcam_Baseline(
        data_dir=args.h5_dataset_path,
        num_frames=args.num_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        width=192,
        height=80,
        indices=args.indices,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-dataset-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./reports/saliency_maps')
    parser.add_argument('--num-frames', type=int, default=12)
    parser.add_argument('--reproduce-paper', action='store_true')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--indices', type=dict, default=None)  # just reserved for later use
    args = parser.parse_args()

    # paper setup
    if args.reproduce_paper:
        args.seed = 11
        args.num_frames = 20
        args.output_dir = './reports/saliency_maps_paper'
        args.indices = {'control': [0, 6, 8, 18], 'tl': [0, 13]}
    else:
        args.indices = {'control': None, 'tl': None}

    # GPU configuration
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('memory growth:', tf.config.experimental.get_memory_growth(device))
    else:
        print('Not enough GPU hardware devices available.')

    main(args)
