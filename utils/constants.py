from torch import FloatTensor

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

LABEL_WEIGHTS = FloatTensor([
        10, #'agriculture',
        5e3, #'artisinal_mine',
        2e3, #'bare_ground',
        2e3, #'blooming',
        5e3, #'blow_down',
        4.5, #'clear',
        50, #'cloudy',
        5e3, #'conventional_mine',
        25, #'cultivation',
        29, #'habitation',
        50, #'haze',
        16, #'partly_cloudy',
        3, #'primary',
        15, #'road',
        2e3, #'selective_logging',
        2e3, #'slash_burn',
        16, #'water'
    ])


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


# THRESHOLDS = [
#         0.245,
#         0.1375,
#         0.2225,
#         0.19,
#         0.0475,
#         0.2375,
#         0.12,
#         0.0875,
#         0.265,
#         0.2175,
#         0.1925,
#         0.1625,
#         0.2625,
#         0.21,
#         0.14,
#         0.085,
#         0.205
#     ]

THRESHOLDS = [0.268406,    
                            0.0144748,     
                            0.0254243,    
                            0.0161414,    
                            0.0150021,    
                            0.7132,
                            0.0306173,     
                            0.0132242,    
                            0.0580758,    
                            0.0433481,    
                            0.0295609,    
                            0.164735,  
                            0.947612,    
                            0.136327,    
                            0.0146465,     
                            0.0143424,    
                            0.089444]



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