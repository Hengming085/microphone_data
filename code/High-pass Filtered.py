import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# 读取音频文件
audio_data, fs = sf.read('D:/my/DARE/MOTOR/Data/5.1：1/6V_0.04A_3.wav')  # 替换为你的音频文件路径

# 如果是立体声，转换为单声道
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)


# 定义Butterworth高通滤波器
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


# 高通滤波器：保留高频
high_cutoff = 5000  # 设置高通滤波的截止频率为5000Hz
high_filtered_data = butter_highpass_filter(audio_data, high_cutoff, fs)


# 频谱分析和峰值处理函数
def plot_fft_with_peaks(data, fs, title="Frequency Spectrum",  prominence_threshold=0.3, distance=1000):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / fs)[:N // 2]
    yf = np.abs(yf[:N // 2])

    # 检测峰值，使用 prominence 参数标记最明显的峰值，并设置最小峰值之间的距离
    peaks, _ = find_peaks(yf, prominence=prominence_threshold, distance=distance)
    peak_freqs = xf[peaks]
    peak_magnitudes = yf[peaks]

    # 绘制频谱图
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(xf, yf, label='Frequency Spectrum')
    plt.plot(peak_freqs, peak_magnitudes, "x", label='Peaks')  # 标记峰值
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()
    plt.show()

    # 输出峰值频率及对应的幅值
    print("检测到的峰值频率及幅度：")
    for freq, mag in zip(peak_freqs, peak_magnitudes):
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")


# 绘制高通滤波后的频谱图，并进行峰值处理
plot_fft_with_peaks(high_filtered_data, fs, "High-pass Filtered (Above 5000 Hz)", prominence_threshold=0.1)
