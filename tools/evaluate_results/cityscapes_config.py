class CityscapesDatasetConfig:
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    
    PATH = "/tmp2/linchiayi/test_exp_env/mmsegmentation/data/cityscapes/gtFine/val"

    SUFFIX = "_gtFine_labelIds.png"

    IGNORE_CLASS = 255

    LABELS = {
        "bicycle": 33,
        "bridge": 15,
        "building": 11,
        "bus": 28,
        "car": 26,
        "caravan": 29,
        "dynamic": 5,
        "ego vehicle": 1,
        "fence": 13,
        "ground": 6,
        "guard rail": 14,
        "motorcycle": 32,
        "out of roi": 3,
        "parking": 9,
        "person": 24,
        "pole": 17,
        "polegroup": 18,
        "rail track": 10,
        "rectification border": 2,
        "rider": 25,
        "road": 7,
        "sidewalk": 8,
        "sky": 23,
        "static": 4,
        "terrain": 22,
        "traffic light": 19,
        "traffic sign": 20,
        "trailer": 30,
        "train": 31,
        "truck": 27,
        "tunnel": 16,
        "unlabeled": 0,
        "vegetation": 21,
        "wall": 12
    }