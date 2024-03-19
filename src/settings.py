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

EEG_GROUPS = [
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

EEG_FEAT_IDX = {
    "Fp1": 0,
    "F3": 1,
    "C3": 2,
    "P3": 3,
    "F7": 4,
    "T3": 5,
    "T5": 6,
    "O1": 7,
    "Fz": 8,
    "Cz": 9,
    "Pz": 10,
    "Fp2": 11,
    "F4": 12,
    "C4": 13,
    "P4": 14,
    "F8": 15,
    "T4": 16,
    "T6": 17,
    "O2": 18,
    "EKG": 19,
}

EEG_GROUP_IDX = [(EEG_FEAT_IDX[a], EEG_FEAT_IDX[b]) for a, b in EEG_GROUPS]
