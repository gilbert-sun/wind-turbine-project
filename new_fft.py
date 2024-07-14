import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.fftpack import fft,ifft

# 从CSV文件中加载数据
data = pd.read_csv(
    "./Meter_time_domain_csv.csv",
    header=0,
)

# 抓時間跟振幅

time=data["Time(s)"].values
amp=data["Amplitude"].values
signal1 = {"time": data["Time(s)"].values, "amp": data["Amplitude"].values}

# 抓頻率跟db
freq=data["Frequency(Hz)"]
db=data["dB"]

# 抓fft
after_fft=fft(amp)




# 進行傅立葉變換，得到頻域數據
def compute_fft(signal):
    freq_domain = np.fft.fft(signal["amp"])
    freq = np.fft.fftfreq(len(signal["time"]))
    positive_freq_indices = np.where(freq > 0)
    positive_freq_domain = freq_domain[positive_freq_indices]
    positive_freq = freq[positive_freq_indices]
    return positive_freq, positive_freq_domain

freq1, freq_domain1 = compute_fft(signal1)


# ! 畫震動規
def plot_true_data():
    plt.subplot(411)
    plt.plot(time, amp)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("time domain")
    plt.grid(True)

    plt.subplot(412)
    plt.plot(freq, db)
    plt.axis([0,100,0,100])#([0,500,0,1700])
    plt.xlabel("Frequence(Hz)")
    plt.ylabel("Pa^2 rms")
    plt.title("Spectrum")
    plt.grid(True)

# ? 畫FFT 
def plot_fft():
    yf0=after_fft[range(int(len(time)/640))]
    yf=abs(after_fft)
    yf1=yf/((len(time)/2)) # 歸一化
    yf2=yf[range(int(len(time)/640))] # 一半區間

    xf=np.arange(len(amp)) # 抓頻率
    xf1=xf
    xf2=xf[range(int(len(time)/640))] # 一半區間

    plt.subplot(413)
    plt.plot(xf2, np.abs(yf2))  # 使用絕對值來得到幅度
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum(FFT)')

    plt.subplot(414)
    plt.plot(freq1, np.abs(freq_domain1 ))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('No ABS Frequency Spectrum(FFT)')


if __name__ == '__main__':

    plot_fft()
    plot_true_data()
    plt.tight_layout() #讓圖整齊
    plt.show()


