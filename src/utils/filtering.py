from scipy.signal import butter, iirnotch, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def notch_filter(data, fs, freq):
    b, a = iirnotch(freq, 30, fs)
    y = filtfilt(b, a, data, axis=0)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs=200, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=1):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def butter_highpass_filter(data, cutoff_freq=0.1, sampling_rate=200, order=1):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
