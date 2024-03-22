import numpy as np
import torch
from scipy.fft import fft, ifft
from sklearn.utils import check_random_state


def _new_random_fft_phase_odd(c, n, random_state):
    rng = check_random_state(random_state)
    random_phase = 2j * np.pi * rng.random((c, (n - 1) // 2))
    return np.concatenate(
        [np.zeros((c, 1)), random_phase, -np.flip(random_phase, [-1])], axis=-1
    )


def _new_random_fft_phase_even(c, n, random_state):
    rng = check_random_state(random_state)
    random_phase = 2j * np.pi * rng.random((c, n // 2 - 1))
    return np.concatenate(
        [
            np.zeros((c, 1)),
            random_phase,
            np.zeros((c, 1)),
            -np.flip(random_phase, [-1]),
        ],
        axis=-1,
    )


_new_random_fft_phase = {0: _new_random_fft_phase_even, 1: _new_random_fft_phase_odd}


def ft_surrogate(X, phase_noise_magnitude, channel_indep=False, random_state=None):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.
    Function copied from https://github.com/cliffordlab/sleep-convolutions-tf
    and modified.
    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    phase_noise_magnitude: float
        Float between 0 and 1 setting the range over which the phase
        perturbation is uniformly sampled:
        [0, `phase_noise_magnitude` * 2 * `pi`].
    channel_indep : bool
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI.
    random_state: int | numpy.random.Generator, optional
        Used to draw the phase perturbation. Defaults to None.
    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    f = fft(X, axis=-1)
    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        f.shape[-2] if channel_indep else 1, n, random_state=random_state
    )
    if not channel_indep:
        random_phase = np.tile(random_phase, (f.shape[0], 1))
    f_shifted = f * np.exp(phase_noise_magnitude * random_phase)
    shifted = ifft(f_shifted, axis=-1)
    transformed_X = shifted.real
    return transformed_X


class SideSwap(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return sample[
                :,
                [
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
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


class FTSurrogate(object):
    def __init__(self, p: float = 0.5, phase_noise_magnitude: float = 0.1):
        self.p = p
        self.phase_noise_magnitude = phase_noise_magnitude

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return ft_surrogate(
                sample,
                self.phase_noise_magnitude,
                channel_indep=False,
                random_state=None,
            )
        return sample
