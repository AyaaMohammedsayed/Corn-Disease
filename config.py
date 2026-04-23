import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(''))
DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'app', 'models')

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
SEED = 123