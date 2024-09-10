import glob,os
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import librosa.display
matplotlib.use("TkAgg")

# Save button
def plot1(event):
    global fs,fl
    fd = os.path.dirname(os.path.abspath(filepath))+"/outdir/"
    if (not os.path.exists(fd)):
        os.makedirs(fd)
    for idx in range(len(fs)):
            png_peak = os.path.basename(fs[idx].split(".wav")[0]+"_out.png")
            y, sr = librosa.load(fs[idx])
            plot221_4(idx, fs, y, sr,fd+png_peak,True)

# To plot previous page
def plot2(event):
    global fs,fl
    fl=fl-1 if fl > 0 else len(fs)-1
    try:
       #     print("prev ------------> " ,fl, fs[fl])
        y, sr = librosa.load(fs[fl])  # sr 為採樣頻率
    except:
        y, sr = librosa.load(fs[0])
        fl = 0
    plot221_4(fl, fs, y, sr)
    plt.show()

# To plot next page
def plot3(event):
    global fs,fl
    fl=fl+1 if fl < len(fs) else 0
    try:
       # print("next------------> " ,fl, fs[fl])
        y, sr = librosa.load(fs[fl])  # sr 為採樣頻率
    except:
        y, sr = librosa.load(fs[0])
        fl = 0

    plot221_4(fl, fs, y, sr)
    plt.show()

#quit
def plot4(event):
    global y,x,p,ax
    quit()


def plot221_4(idx, fs, y, sr, fout=None,savefig = False):
    # =======================================221
    plt.subplot(221)
    plt.cla()  # ax0.clear()
    sound, _ = librosa.effects.trim(y)  # trim silent edges
    plt.title('Machine sound: ' + fs[idx].split("/")[-1])
    plt.ylabel('Ampitude')
    plt.xlabel('Hz')
    plt.grid()
    librosa.display.waveshow(y=sound, sr=sr)  # .waveplot(sound, sr=sr)
    # =======================================222
    ax1 = plt.subplot(222)
    plt.cla()  # ax1.clear()
    n_fft = 32800
    fft_signal = np.abs(librosa.stft(sound[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    plt.title("FFT for sound")
    plt.xlabel('Hz')
    # plt.tight_layout()
    plt.xlim(0,5000)
    plt.ylim(0,1)
    plt.grid()
    plt.plot(fft_signal)
    # =======================================223
    plt.subplot(223)
    plt.cla()
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64, power=2.0)
    img = librosa.display.specshow(librosa.power_to_db(mel_spect), sr=sr, x_axis='time', y_axis='mel', fmax=16400)
    plt.title('Mel spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Freq [Hz]')
    # plt.colorbar(img, format='%+2.0f dB')
    # =======================================224
    plt.subplot(224)
    plt.cla()
    plt.title('Spectrogram')
    plt.xlabel('Time [s]')
    Pxx, freqs, times, cax = plt.specgram(y, Fs=sr,mode='magnitude', scale='dB')
    #plt.gca().set_ylim(0,16384)
    #plt.colorbar( cax, format='%+2.0f dB')
    if savefig:
        plt.savefig(fout)  # os.path.join(savdir, savname+ ".png"))


def plot_4button():
    global btn1, btn2, btn3, btn4, axButn1, axButn2, axButn3, axButn4
    # Save button
    axButn1 = plt.axes([0.35, 0.02, 0.04, 0.02])
    btn1 = Button(axButn1, label="Save", color='pink', hovercolor='lime')  # , useblit=True)
    btn1.on_clicked(plot1)
    # Previous button
    axButn2 = plt.axes([0.4, 0.02, 0.04, 0.02])
    btn2 = Button(axButn2, label="Previous", color='pink', hovercolor='lime')  # , useblit=True)
    btn2.on_clicked(plot2)
    # Next button
    axButn3 = plt.axes([0.45, 0.02, 0.04, 0.02])
    btn3 = Button(axButn3, label="Next", color='pink', hovercolor='lime')  # , useblit=True)
    btn3.on_clicked(plot3)
    axButn4 = plt.axes([0.5, 0.02, 0.04, 0.02])
    btn4 = Button(axButn4, label="Quit", color='cyan', hovercolor='yellow')  # , useblit=True)
    btn4.on_clicked(plot4)
    # plt.subplots_adjust(bottom=0.05)


if __name__ == '__main__':
    global btn1, btn2, btn3, btn4, axButn1, axButn2, axButn3, axButn4
    global fs,fl
    rows = 2
    cols = 2
    
    # Generate indices for the grid
    plt.figure(figsize=(18, 9))
    fd = filedialog.askdirectory()
    fs = glob.glob(fd+"/**/*.wav", recursive=True)
    fs.sort()
    fl = 0 
    filepath = fs[0]
    
    #print("-----fidx ---> " , fl  ,"========> ",filepath )#,"========> ",filepath.split(".")[0].split('/'))
    y, sr = librosa.load(filepath)  # sr 為採樣頻率
    plot221_4(fl, fs, y, sr)
    plot_4button()
    plt.show()
