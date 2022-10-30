import numpy as np
import librosa
import soundfile


def load_samples(filename, sr=None, offset=None, duration=None):
    """Loads audio samples from file.

    Args:
        filename (str): Path to audio file.
        sr (int, optional): Sample rate to resample audio. If None sample rate isn't changed. Defaults to None.
        offset (float, optional): Time in seconds from which Loading is started. If None offset equals 0. Defaults to None.
        duration (float, optional): Duration in seconds to load from audio file. If None duration equals audio file length. Defaults to None.

    Returns:
        (ndarray, int): Tuple of 1D array of samples and value of sample rate.
    """
    samples, sample_rate = librosa.load(
        filename, sr=sr, mono=True, offset=offset, duration=duration)
    return samples, sample_rate


def spectrogram(samples, sample_rate, fft, hop_length, window, max_freq=None):
    """Converts audio samples to spectrogram.

    Args:
        samples (ndarray): 1D array of samples.
        sample_rate (float): Sample rate.
        fft (int): FFT frame width in seconds.
        hop_length (int): Hop length in seconds.
        window (str): Window function to apply on frame.
        max_freq (int, optional): Maximal frequency to cut spectrogram values. If None do not cut. Defaults to None.

    Returns:
        ndarray: 2D array of spectrogram values.
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(
        samples, n_fft=int(fft * sample_rate), hop_length=int(hop_length * sample_rate), window=window)))
    if max_freq is not None:
        bins_amount = int(len(D) * max_freq / (sample_rate / 2))
        D = D[:bins_amount, :]
    return D


def normalized_spectrogram(D, audio_mean, audio_std):
    """Normalize spectrogram.

    Args:
        D (ndarray): 2D array of spectrogram values.
        audio_mean (float): Mean of spectrogram values used to normalize.
        audio_std (float): Standard deviation of spectrogram values used to normalize.

    Returns:
        ndarray: 2D array of normalized spectrogram values.
    """
    D -= audio_mean
    D /= audio_std
    return D


def delta_spectrogram(D):
    """Get delta spectrogram.

    Args:
        D (ndarray): 2D array of spectrogram values.

    Returns:
        ndarray: 2D array of delta spectrogram values.
    """
    return librosa.feature.delta(D)


def get_normalized_spectrogram(filename, config, offset=None, duration=None):
    """Load audio file and convert to normalized spectrogram.

    Args:
        filename (str): Path to audio file.
        config (dict): Config parameters to calculate spectrogram.
        offset (float, optional): Time in seconds from which loading is started. If None offset equals 0. Defaults to None.
        duration (float, optional): Duration in seconds to load from audio file. If None duration equals audio file length. Defaults to None.

    Returns:
        ndarray: 2D array of normalized spectrogram values.
    """
    samples, sample_rate = load_samples(
        filename, offset=offset, duration=duration)
    spec = spectrogram(samples, sample_rate,
                       config['fft'], config['hop_length'], config['window_type'], config['max_freq'])
    return normalized_spectrogram(spec, config['audio_mean'], config['audio_std'])


def split_spectrogram_to_windows(spec, length, overlapping, chunk_length):
    """Split spectrogram into frames of given width.

    Args:
        spec (ndarray): 2D array of spectrogram values.
        length (float): Length of audio that spectrogram represents in seconds.
        overlapping (float): Overlapping of divided frames. 1 - no overlapping, 2 - 50% overlapping, 4 - 75% overlapping. 
        chunk_length (float): 

    Returns:
        list[ndarray]: List of 2D arrays of framed spectrogram values.
    """
    size = int(length * overlapping /
               chunk_length - overlapping + 1)
    step = int(chunk_length * spec.shape[1] / length)
    split_spectrogram = [spec[:, min(spec.shape[1], int(i * spec.shape[1] / size) + step) - step:min(
        spec.shape[1], int(i * spec.shape[1] / size) + step)] for i in range(size)]
    return split_spectrogram


def save_wav(filename, samples, sample_rate):
    """Save audio to file from samples

    Args:
        filename (str): Path to audio file to save.
        samples (ndarray): 1D array of samples.
        sample_rate (int): Sample rate.
    """
    soundfile.write(filename, samples, sample_rate)


def get_wav_length(filename):
    """Gets length of audio file.

    Args:
        filename (str): Path to audio file.

    Returns:
        float: Length of audio file in seconds.
    """
    return librosa.get_duration(filename=filename)


def noise(shape, noise_type):
    if noise_type == 'gaussian':
        return gaussian(shape, 0.1)
    return np.zeros(shape)


def gaussian(shape, std):
    return np.random.normal(0, std, shape)
