class CamVidDatasetConfig:
    CLASSES = ('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian',
               'Road', 'Sidewalk', 'SignSymbol', 'Sky', 'Tree')

    PALETTE = [[0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
               [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
               [192, 128, 128], [128, 128, 128], [128, 128, 0]]
    
    PATH = "/tmp2/linchiayi/test_exp_env/mmsegmentation/data/CamVid/val_labelIds"

    SUFFIX = "_L.png"

    IGNORE_CLASS = 255