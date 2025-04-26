import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt, find_peaks, wiener
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt

# 读取音频文件
audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/136new/inside/3.5V_001.wav', sr=None)

# Butterworth低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# def butter_lowpass_filter(data, cutoff, fs, order=10):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
#     return sosfiltfilt(sos, data)

# 应用低通滤波
low_cutoff = 5
filtered_data = butter_lowpass_filter(audio_data, low_cutoff, fs)

# 使用维纳滤波进行降噪处理
denoised_data = wiener(filtered_data, mysize=513)


# 时域波形绘制函数，可选任意时间区间
def plot_time_domain(data, fs, title="Time Domain Signal", start_time=5.0, end_time=10.0):
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    t = np.arange(start_sample, end_sample) / fs
    segment = data[start_sample:end_sample]

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(t, segment, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

# 显示5-10秒内的降噪后的时域信号
plot_time_domain(denoised_data, fs, title="Denoised Time Domain Signal", start_time=10.0, end_time=12.0)

# 计算傅里叶变换并归一化

def compute_fft(data, fs):
    N = len(data)
    freq_spectrum = fft(data)
    amplitude_spectrum = np.abs(freq_spectrum[:N // 2])
    amplitude_spectrum /= np.max(amplitude_spectrum)
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, amplitude_spectrum

# 频谱分析及峰值检测函数

def plot_fft_with_peaks(data, fs, title="Frequency Spectrum", threshold=0.1, distance=1000):
    freqs, amplitude_spectrum = compute_fft(data, fs)

    peaks, _ = find_peaks(amplitude_spectrum, prominence=threshold, distance=distance)
    peak_freqs = freqs[peaks]
    peak_magnitudes = amplitude_spectrum[peaks]

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(freqs, amplitude_spectrum, label='Normalized Amplitude Spectrum')
    plt.plot(peak_freqs, peak_magnitudes, "x", label='Detected Peaks')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.title(title)
    plt.xlim(0, 5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Detected Peak Frequencies:")
    for freq, mag in zip(peak_freqs, peak_magnitudes):
        print(f"Frequency: {freq:.2f} Hz, Amplitude: {mag:.2f}")

# 绘制降噪后频谱图
plot_fft_with_peaks(denoised_data, fs, "Denoised Frequency Spectrum", threshold=0.1)
