import argparse
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.signal import filtfilt, iirnotch
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from scipy.signal import filtfilt as scipy_filtfilt, butter as scipy_butter
from tqdm.auto import tqdm


def create_spectrogram_with_cupy(
    eeg_data,
    start,
    duration=50,
    low_cut_freq=0.2,
    high_cut_freq=40,
    order_band=5,
    spec_size_freq=267,
    spec_size_time=30,
    nperseg=1500,
    noverlap=1483,
    nfft=2750,
    sigma_gaussian=0.7,
    mean_montage_names=4,
):
    electrode_pair_name_locations = {
        "LL": ["Fp1", "F7", "T3", "T5", "O1"],
        "RL": ["Fp2", "F8", "T4", "T6", "O2"],
        "LP": ["Fp1", "F3", "C3", "P3", "O1"],
        "RP": ["Fp2", "F4", "C4", "P4", "O2"],
    }

    # Filter specifications
    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    # Bandpass and notch filter
    # bandpass_coefficients = butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)
    sci_bandpass_coefficients = scipy_butter(
        order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype="band"
    )

    spec_size = duration * 200
    start = start * 200
    real_start = int(start + (10_000 // 2) - (spec_size // 2))
    eeg_data = eeg_data.iloc[real_start : real_start + spec_size]

    # Spectrogram parameters
    fs = 200

    if spec_size_freq <= 0 or spec_size_time <= 0:
        freq_size = int((nfft // 2) / 5.15198) + 1
        segments = int((spec_size - noverlap) / (nperseg - noverlap))
    else:
        freq_size = spec_size_freq
        segments = spec_size_time

    # Initialize spectrogram container
    spectrogram = cp.zeros((freq_size, segments, 4), dtype="float32")

    processed_eeg = {}

    for i, (electrode_pair_name, electrode_locs) in enumerate(
        electrode_pair_name_locations.items()
    ):
        processed_eeg[electrode_pair_name] = np.zeros(spec_size)

        for j in range(4):
            # Compute differential signals
            signal = cp.array(
                eeg_data[electrode_locs[j]].values
                - eeg_data[electrode_locs[j + 1]].values
            )

            # Handles NaNs
            mean_signal = cp.nanmean(signal)
            signal = (
                cp.nan_to_num(signal, nan=mean_signal)
                if cp.isnan(signal).mean() < 1
                else cp.zeros_like(signal)
            )

            # Filters bandpass and notch
            signal_filtered = filtfilt(*notch_coefficients, signal)
            signal_filtered = scipy_filtfilt(
                *sci_bandpass_coefficients, signal_filtered.get()
            )  # HOTFIX

            # GPU-accelerated spectrogram computation
            frequencies, times, Sxx = cupyx_spectrogram(
                signal_filtered, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft
            )

            # Filters frequency range
            valid_freq = (frequencies >= 0.59) & (frequencies <= 20)
            Sxx_filtered = Sxx[valid_freq, :]

            # Logarithmic transformation and normalization using Cupy
            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (
                std + normalization_epsilon
            )

            spectrogram[:, :, i] += spectrogram_slice
            processed_eeg[f"{electrode_locs[j]}_{electrode_locs[j + 1]}"] = signal.get()
            processed_eeg[electrode_pair_name] += signal.get()

        # AVERAGES THE 4 MONTAGE DIFFERENCES
        if mean_montage_names > 0:
            spectrogram[:, :, i] /= mean_montage_names

    # Applies Gaussian filter and retrieves the spectrogram as a NumPy array using cupy.ndarray.get()
    spec_numpy = (
        gaussian_filter(spectrogram, sigma=sigma_gaussian).get()
        if sigma_gaussian > 0
        else spectrogram.get()
    )

    # Filter EKG signal
    ekg_signal_filtered = filtfilt(
        *notch_coefficients, cp.array(eeg_data["EKG"].values)
    )
    processed_eeg["EKG"] = scipy_filtfilt(
        *sci_bandpass_coefficients, ekg_signal_filtered.get()
    )  # HOTFIX
    return spec_numpy, processed_eeg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the EEG data file in CSV format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output folder where spectrogram will be saved.",
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to use the training data."
    )

    args = parser.parse_args()
    csv_file = "train.csv" if args.train else "test.csv"
    eeg_folder = "train_eegs" if args.train else "test_eegs"

    df = pd.read_csv(Path(args.input) / csv_file)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows()):
        eeg_id = row["eeg_id"]
        offset = int(row["eeg_label_offset_seconds"])
        eeg_data = pd.read_parquet(f"{Path(args.input)}/{eeg_folder}/{eeg_id}.parquet")
        spec_numpy, processed_eeg = create_spectrogram_with_cupy(
            eeg_data,
            start=offset,
            duration=50,
            low_cut_freq=0.7,
            high_cut_freq=20,
            order_band=5,
            spec_size_freq=267,
            spec_size_time=501,
            nperseg=1500,
            noverlap=1483,
            nfft=2750,
            sigma_gaussian=0.0,
            mean_montage_names=4,
        )
        np.save(str(Path(args.output) / f"{eeg_id}_{offset}.npy"), spec_numpy)
