import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# 读取音频文件
audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/136new/3.5V_003.wav', sr=None)  # 替换为你的音频文件路径


# 定义Butterworth低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist 频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# 低通滤波：保留低频
low_cutoff = 20  # 设置低通滤波的截止频率
low_filtered_data = butter_lowpass_filter(audio_data, low_cutoff, fs)


# 计算傅里叶变换
def compute_fft(data, fs):
    N = len(data)
    freq_spectrum = fft(data)
    amplitude_spectrum = np.abs(freq_spectrum[:N // 2])
    amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)  # 归一化幅值
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, amplitude_spectrum


# 频谱分析及峰值检测
def plot_fft_with_peaks(data, fs, title="Frequency Spectrum", threshold=0.1, distance=1000):
    freqs, amplitude_spectrum = compute_fft(data, fs)

    # 检测峰值
    peaks, _ = find_peaks(amplitude_spectrum, prominence=threshold, distance=distance)
    peak_freqs = freqs[peaks]
    peak_magnitudes = amplitude_spectrum[peaks]

    # 绘制频谱图
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(freqs, amplitude_spectrum, label='Normalized Amplitude Spectrum')
    plt.plot(peak_freqs, peak_magnitudes, "x", label='Detected Peaks')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.title(title)
    plt.xlim(0, 20)  # 限制横轴范围
    plt.grid()
    plt.legend()
    plt.show()

    # 输出检测到的峰值
    print("Detected Peak Frequencies:")
    for freq, mag in zip(peak_freqs, peak_magnitudes):
        print(f"Frequency: {freq:.2f} Hz, Amplitude: {mag:.2f}")


# 进行频谱分析并检测峰值
plot_fft_with_peaks(low_filtered_data, fs, "Low-pass Filtered Frequency Spectrum", threshold=0.1)