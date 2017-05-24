NUM_CLASSES = 17

NUM_TRAIN = 40479
NUM_TEST= 40669

IMG_W = None
IMG_H = None

TRAIN_DATA_PATH = "./input/train-jpg/"
TEST_DATA_PATH = './input/test-jpg/'
TRAIN_LABELS_PATH = "./input/train_v2.csv"
TEST_LABELS_PATH = "./input/sample_submission.csv"


BEST_MODEL_FILENAME = 'best_model.ckpt'

# TODO: double check list
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

LABEL_LIST = ['agriculture',
        'artisinal_mine',
        'bare_ground',
        'blooming',
        'blow_down',
        'clear',
        'cloudy',
        'conventional_mine',
        'cultivation',
        'habitation',
        'haze',
        'partly_cloudy',
        'primary',
        'road',
        'selective_logging',
        'slash_burn',
        'water']