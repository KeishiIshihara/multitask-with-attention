#=============================================================
# This code is inteded to run a trained agent on some basic
# experiment scenarios to see how the agents drive in carla.
#
# This code is adapted from:
#   https://github.com/carla-simulator/driving-benchmarks/blob/master/benchmarks_084.py
#
#=============================================================


"""
Command example:
```console
$ python enjoy.py \
    --gpu 0 \
    --port 2000 \
    --agent MTA \
    --city-name Town01 \
    --verbose \
    --continue-experiment
```
"""

import argparse
import logging
import os
import time
import tensorflow as tf
from pathlib import Path

from driving_benchmarks.version084.benchmark_tools import run_driving_benchmark
from driving_benchmarks.version084.driving_benchmarks import BasicExperimentSuite
from model.agents import BaselineAgent, MTAAgent, CILRSAgent, MTAgent


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print extra status information at every timestep')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log-name',
        metavar='T',
        default='enjoy',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--log-path',
        metavar='PATH',
        default='/tmp/multitask_with_attention/enjoy/',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )
    argparser.add_argument(
        '-g', '--gpu',
        default='0',
        type=str,
        help='GPU ID'
    )
    argparser.add_argument(
        '-a', '--agent',
        default='MTA',
        choices=['baseline', 'MTA', 'CILRS', 'MT'],
        help='Agent to test'
    )
    argparser.add_argument(
        '--record',
        action='store_true',
        help='Record rendered information as mp4 video'
    )
    args = argparser.parse_args()


    # Logging config
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            logging.info('memory growth:', tf.config.experimental.get_memory_growth(device))
    else:
        logging.info('Not enough GPU hardware devices available')

    # Configs
    Path(args.log_path).mkdir(exist_ok=True)
    video_log_dir = str(Path(args.log_path) / 'video')
    has_display = True
    seq_len = 1
    camera_size = (160, 384, 3)

    # Initialize agent
    if args.agent == 'baseline':
        weight_path = 'ckpts/baseline/ckpt'
        agent = BaselineAgent(camera_size, weight_path, seq_len,
                              has_display, args.record, video_log_dir)
    elif args.agent == 'MTA':
        weight_path = 'ckpts/MTA/ckpt'
        agent = MTAAgent(camera_size, weight_path, seq_len,
                         has_display, args.record, video_log_dir)
    elif args.agent == 'CILRS':
        weight_path = 'ckpts/CILRS/ckpt'
        agent = CILRSAgent(camera_size, weight_path,
                           has_display, args.record, video_log_dir)
    else:
        weight_path = 'ckpts/MT/ckpt'
        agent = MTAgent(camera_size, weight_path,
                        has_display, args.record, video_log_dir)

    # Build experiment suit and run experiment
    experiment_suite = BasicExperimentSuite(args.city_name)
    enter = time.time()
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.log_path, args.continue_experiment,
                          args.host, args.port)
    end = time.time()
    print(f'Time taken: {end - enter:.0f} s')

