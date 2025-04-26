# import numpy as np
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# from scipy.signal import butter, filtfilt
#
# # 读取音频文件
# audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/700：1_assembled/Fully_Assembled_13.wav', sr=None)  # 替换为你的音频文件路径
#
# # 定义Butterworth低通滤波器
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     nyq = 0.5 * fs  # Nyquist 频率
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, data)
#
# # 低通滤波：保留低频
# low_cutoff = 5  # 设置低通滤波的截止频率
# low_filtered_data = butter_lowpass_filter(audio_data, low_cutoff, fs)
#
# # 计算梅尔频谱
# def compute_mel_spectrogram(data, fs, n_mels=128):
#     mel_spec = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=n_mels)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为dB单位
#     return mel_spec_db
#
# # 绘制梅尔频谱图
# mel_spectrogram = compute_mel_spectrogram(low_filtered_data, fs)
# plt.figure(figsize=(10, 4), dpi=150)
# librosa.display.specshow(mel_spectrogram, sr=fs, x_axis='time', y_axis='mel')
# plt.colorbar(label='Decibels (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Mel Scale)")
# plt.ylim(0, 5)  # 限制纵轴范围为0-100
# plt.title("Mel Spectrogram Representation of Filtered Audio")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, filtfilt

# 读取音频文件
audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/700：1_assembled/Fully_Assembled_13.wav', sr=None)  # 替换为你的音频文件路径

# 定义Butterworth低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist 频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 低通滤波：保留低频
low_cutoff = 20  # 设置低通滤波的截止频率
low_filtered_data = butter_lowpass_filter(audio_data, low_cutoff, fs)

# 计算短时傅里叶变换（STFT）
def compute_stft(data, fs, n_fft=1024, hop_length=512):
    stft_result = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    return stft_result, librosa.fft_frequencies(sr=fs, n_fft=n_fft), librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=fs, hop_length=hop_length)

# 绘制频谱图（频率轴使用 Hz 而非 Mel Scale）
stft_result, freq_axis, time_axis = compute_stft(low_filtered_data, fs)
plt.figure(figsize=(10, 4), dpi=150)
librosa.display.specshow(librosa.amplitude_to_db(stft_result, ref=np.max), sr=fs, x_axis='time', y_axis='linear')
plt.colorbar(label='Decibels (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("STFT Spectrogram Representation of Filtered Audio")
plt.ylim(0, 20)  # 限制纵轴频率范围
plt.show()
