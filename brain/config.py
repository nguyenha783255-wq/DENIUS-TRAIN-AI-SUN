import os

SOURCE_URL = os.getenv(
    'SOURCE_URL',
    'https://kwinstore.com/sunwin/tx/history/b54b32ca9f748d5dbe64f421f14f1f04fa8d30012b17d0f5'
)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/denius_ai')
POLL_SECONDS = int(os.getenv('POLL_SECONDS', '15'))
RETRAIN_EVERY_NEW = int(os.getenv('RETRAIN_EVERY_NEW', '100'))
MIN_TRAIN_ROWS = int(os.getenv('MIN_TRAIN_ROWS', '500'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '20260917'))
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '20'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'DENIUS-BRAIN-v1')
