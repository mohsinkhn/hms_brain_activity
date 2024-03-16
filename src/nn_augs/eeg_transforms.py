import numpy as np


class SideSwap(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return sample[
                :,
                [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    8,
                    9,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    18,
                    19,
                    20,
                    21,
                    22,
                ],
            ]
        return sample


class HorizontalFlip(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return sample[::-1]
        return sample


class SignFlip(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return sample * -1
        return sample


class Stretch(object):
    def __init__(self, p: float = 0.5, max_stretch: int = 0.1):
        self.p = p
        self.max_stretch = max_stretch

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            drop_num = int(len(sample) * np.random.rand() * self.max_stretch)
            # interpolate after removing strech %
            return np.interp(
                np.arange(0, sample.shape[0] - drop_num, 1),
                np.arange(0, sample.shape[0], 1),
                sample[:-drop_num],
            )
        return sample


class EdgeMasking(object):
    def __init__(self, p: float = 0.5, max_mask: int = 0.2):
        self.p = p
        self.max_mask = max_mask

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            mask_num = int(len(sample) * np.random.rand() * self.max_mask)
            if np.random.rand() < 0.5:
                sample[-mask_num:] = 0
            else:
                sample[:mask_num] = 0
            return sample
        return sample


class Roll(object):
    def __init__(self, p: float = 0.5, max_roll: int = 0.1):
        self.p = p
        self.max_roll = max_roll

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            roll_num = int(len(sample) * np.random.rand() * self.max_roll)
            return np.roll(sample, roll_num)
        return sample


class AmplitudeChange(object):
    def __init__(self, p: float = 0.5, max_zoom: int = 0.1):
        self.p = p
        self.max_zoom = max_zoom

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            zoom = 1 + (np.random.rand() - 0.5) * self.max_zoom
            return sample * zoom
        return sample


class GaussianNoise(object):
    def __init__(self, p: float = 0.5, max_noise: int = 0.1):
        self.p = p
        self.max_noise = max_noise

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            noise = np.random.randn(*sample.shape) * self.max_noise
            return sample + noise * sample
        return sample
