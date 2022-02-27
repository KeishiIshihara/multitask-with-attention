import numpy as np


TL = {0:'GREEN', 1:'YELLOW', 2:'RED', 3:'NONE'}
COMMAND_CONVERTER = {
    0.0: 0,
    1.0: 0,
    2.0: 0,
    3.0: 1,
    4.0: 2,
    5.0: 3,
}
COMMAND = {
    0: 'GOAL',
    2: 'FOLLOW',
    3: 'LEFT',
    4: 'RIGHT',
    5: 'STRAIGHT'
}
cityscapes_palette = {
    0: (  0,  0,  0),  # Unlabeled
    1: ( 70, 70, 70),  # Building
    2: (190,153,153),  # Fence
    3: ( 72,  0, 90),  # Others
    4: (220, 20, 60),  # Pedestrian
    5: (153,153,153),  # Pole
    6: (157,234, 50),  # Road line
    7: (128, 64,128),  # Road
    8: (244, 35,232),  # Sidewalk
    9: (107,142, 35),  # Vegetation
    10: (  0,  0,255), # Car
    11: (102,102,156), # Wall
    12: (220,220,  0), # Traffic sign
}
# COLOR_cityscapes = np.uint8(list(cityscapes_palette.values()))
COLOR_cityscapes = np.uint8([
    (  0,  0,  0),  # Unlabeled
    ( 70, 70, 70),  # Building
    (190,153,153),  # Fence
    ( 72,  0, 90),  # Others
    (220, 20, 60),  # Pedestrian
    (153,153,153),  # Pole
    (157,234, 50),  # Road line
    (128, 64,128),  # Road
    (244, 35,232),  # Sidewalk
    (107,142, 35),  # Vegetation
    (  0,  0,255), # Car
    (102,102,156), # Wall
    (220,220,  0), # Traffic sign
])
# cityscapes (traffic sign  to traffic light)
CONVERTER_cityscapes = np.uint8([
    0,    # unlabeled
    1,    # building
    2,    # fence
    3,    # others
    4,    # ped
    5,    # pole
    6,    # road line
    7,    # road
    8,    # sidewalk
    9,    # vegetation
    10,    # car
    11,    # wall
    12,    # traffic sign
])

