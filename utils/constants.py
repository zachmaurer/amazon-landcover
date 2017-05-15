NUM_CLASSES = 17

NUM_TRAIN = None
NUM_VAL = None
NUM_TEST= None

TRAIN_DATA_PATH = './input/train-jpg/'
TEST_DATA_PATH = './input/test-jpg/'
TRAIN_LABELS_PATH ='./input/train_v2.csv' 


# TODO: double check
LABEL_MAP = {
        0: 'agriculture',
        1: 'artisinal_mine',
        2: 'bare_ground',
        3: 'blooming',
        4: 'blow_down',
        5: 'clear',
        6: 'cloudy',
        7: 'conventional_mine',
        8: 'cultivation',
        9: 'habitation',
        10: 'haze',
        11: 'partly_cloudy',
        12: 'primary',
        13: 'road',
        14: 'selective_logging',
        15: 'slash_burn',
        16: 'water'
    }