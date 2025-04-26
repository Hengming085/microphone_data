import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import datetime  # 用于生成时间戳
import os

# 设置采样率和录制时长
fs = 44100  # 采样率（每秒采样点数）
duration = 15  # 录制时间（秒）

# 从麦克风录制音频
print("正在录制中...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # 等待录音结束
print("录制完成")

# 将音频数据转换为一维数组
audio_data = audio_data.flatten()

# 使用当前时间戳生成唯一文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
audio_file_name = f'recorded_audio_{timestamp}.wav'
image_file_name = f'waveform_{timestamp}.png'  # 图片文件名与音频文件名一致

# 选择保存路径
save_dir = 'D:/my/DARE/MOTOR/Data/136new/inside'  # 替换为你希望保存音频的路径
# if not os.path.exists(save_dir):
  # os.makedirs(save_dir)  # 如果目录不存在，创建该目录

# 拼接保存路径和文件名
audio_file_path = os.path.join(save_dir, audio_file_name)
image_file_path = os.path.join(save_dir, image_file_name)

# 保存录制的音频为 WAV 文件
sf.write(audio_file_path, audio_data, fs)
print(f"音频已保存为 {audio_file_path}")

# 方法1：增加分辨率（DPI）和调整图像大小
plt.figure(figsize=(15, 4), dpi=150)  # 图像大小为10x4，DPI设置为150

# 方法2：减少数据点（下采样）
downsample_factor = 10  # 调整下采样因子，减少绘制的数据点
downsampled_data = audio_data[::downsample_factor]
downsampled_time = np.linspace(0, duration, len(downsampled_data))

# 方法3：绘制波形图，使用透明度和调整线条宽度
plt.plot(downsampled_time, downsampled_data, alpha=0.7, linewidth=0.8)  # 设置透明度和线条宽度
plt.title("Audio Signal Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 方法4：放大特定区域（可选，显示2到4秒的部分）
plt.xlim(2, 4)  # 放大2到4秒的波形

# 保存图片
plt.savefig(image_file_path)  # 保存图像为 PNG 文件
print(f"图片已保存为 {image_file_path}")

# 显示图像
plt.show()
