DATA_ROOT = "./data"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"

SAMPLE_RATE = 200
EEG_DURATION = 50  # seconds
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

TRAIN_EEGS = f"{DATA_ROOT}/train_eegs"
TEST_EEGS = f"{DATA_ROOT}/test_eegs"

TRAIN_SPECTOGRAMS = f"{DATA_ROOT}/train_spectrograms"
TEST_SPECTOGRAMS = f"{DATA_ROOT}/test_spectrograms"
