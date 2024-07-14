import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Button


# 加載CSV文件的函數
def load_csv_file():
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    filepath = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    return filepath


# 選擇CSV文件
csv_file = load_csv_file()

# 從CSV文件中加載數據
data = pd.read_csv(csv_file, header=1)

# 提取時間和振幅
signals = {
    "red_x": {"time": data["Red ROI - XT"], "amp": data["Red ROI - XA"]},
    "blue_x": {"time": data["Blue ROI - XT"], "amp": data["Blue ROI - XA"]},
    "green_x": {"time": data["Green ROI - XT"], "amp": data["Green ROI - XA"]},
    "pink2_x": {"time": data["Pink2 ROI - XT"], "amp": data["Pink2 ROI - XA"]},
    "purple2_x": {"time": data["Purple2 ROI - XT"], "amp": data["Purple2 ROI - XA"]},
}


# 進行傅立葉變換，得到頻域數據
def compute_fft(signal):
    freq_domain = np.fft.fft(signal["amp"])
    freq = np.fft.fftfreq(len(signal["time"]), d=signal["time"][1] - signal["time"][0])
    positive_freq_indices = np.where(freq > 0)
    positive_freq_domain = freq_domain[positive_freq_indices]
    positive_freq = freq[positive_freq_indices]
    return positive_freq, positive_freq_domain


# 進行STFT計算
def compute_stft(signal):
    y = np.array(signal["amp"])  # 將音訊數據轉換為 numpy.ndarray 類型
    sr = 1 / (signal["time"][1] - signal["time"][0])  # 計算採樣率
    stft_data = librosa.stft(y)
    return stft_data, sr


# 計算並繪製梅爾頻譜圖
def plot_mel_spectrograms(signals):
    plt.figure(figsize=(12, 10))
    for i, (key, signal) in enumerate(signals.items()):
        y = np.array(signal["amp"])  # 將音訊數據轉換為 numpy.ndarray 類型
        sr = 1 / (signal["time"][1] - signal["time"][0])  # 計算採樣率
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        plt.subplot(5, 1, i + 1)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f'{key.replace("_", " ").title()} Mel Spectrogram')
        plt.tight_layout()
    plt.show()


# 繪製所有數據的時域圖
def plot_all_data(signals):
    plt.figure(figsize=(12, 10))
    for i, (key, signal) in enumerate(signals.items()):
        plt.subplot(5, 1, i + 1)
        plt.plot(signal["time"], signal["amp"])
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(key.replace("_", " ").title())
        plt.grid(True)
        plt.tight_layout()
    plt.show()


# 繪製頻域圖
def plot_fft(signals):
    plt.figure(figsize=(12, 10))
    for i, (key, signal) in enumerate(signals.items()):
        freq, freq_domain = compute_fft(signal)
        plt.subplot(5, 1, i + 1)
        plt.plot(freq, np.abs(freq_domain))
        plt.title(f"{key.replace('_', ' ').title()} Frequency Domain")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
    plt.show()


# 創建按鈕並綁定操作
def on_time_domain_button_clicked(event):
    plot_all_data(signals)


def on_frequency_domain_button_clicked(event):
    plot_fft(signals)


def on_mel_spectrogram_button_clicked(event):
    plot_mel_spectrograms(signals)


def on_show_all_button_clicked(event):
    plot_all_data(signals)
    plot_fft(signals)
    plot_mel_spectrograms(signals)


# 創建一個matplotlib圖形並添加按鈕
plt.subplots_adjust(bottom=0.4)

# 添加按鈕
ax_time_domain = plt.axes([0.35, 0.8, 0.3, 0.075])
ax_frequency_domain = plt.axes([0.35, 0.6, 0.3, 0.075])
ax_mel_spectrogram = plt.axes([0.35, 0.4, 0.3, 0.075])
ax_show_all = plt.axes([0.35, 0.2, 0.3, 0.075])

btn_time_domain = Button(ax_time_domain, "Time Domain")
btn_frequency_domain = Button(ax_frequency_domain, "Frequency Domain")
btn_mel_spectrogram = Button(ax_mel_spectrogram, "Mel Spectrogram")
btn_show_all = Button(ax_show_all, "Show All")
# 綁定按鈕操作
btn_time_domain.on_clicked(on_time_domain_button_clicked)
btn_frequency_domain.on_clicked(on_frequency_domain_button_clicked)
btn_show_all.on_clicked(on_show_all_button_clicked)
btn_mel_spectrogram.on_clicked(on_mel_spectrogram_button_clicked)

plt.show()
