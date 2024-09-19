import glob,os
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import librosa.display
import matplotlib.patches as mpatches
matplotlib.use("TkAgg")

# Save button
def plot1(event):
    global fs,fl,fdd
    fd = fdd+"/outdir/" #os.path.dirname(os.path.abspath(filepath))+"/outdir/"
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
    plt.xlim(0,10)
    plt.grid()
    librosa.display.waveshow(y=sound, sr=sr)  # .waveplot(sound, sr=sr)
    # =======================================222
    ax1 = plt.subplot(222)
    plt.cla()  # ax1.clear()
    n_fft = 40960
    fft_signal = np.abs(librosa.stft(sound[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    plt.title("FFT for sound")
    plt.xlabel('Hz')
    # plt.tight_layout()
    #plt.xlim(0,5000)
    plt.gca().set_xlim(0,1000)
    plt.ylim(0,1)
    plt.grid()
    plt.plot(fft_signal)
    # =======================================2
    # 偵測節拍
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(tempo,  " <<: :>> " ,beat_frames)
    # 將 frames 轉為實際時間
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # 節拍的時間點
    print(len(beat_times), " <<===>> " , beat_times)
    print("\n--------------------------------------------------")
    plt.subplot(223)

    plt.cla() #n_fft = 1024 , hop_length = 512/ 320
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=40960, hop_length=512, n_mels=128, power=2.0)#,
    img = librosa.display.specshow(librosa.power_to_db(mel_spect, ref=np.max), x_axis='time', y_axis='mel', fmax=16400)#,cmap ="Greys")
    #img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='mel', x_axis='time', fmax=16400)#, ax=ax)
    plt.title('Mel spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Freq [Hz]')
    plt.xlim(0,10)

    # =======================================224
    plt.subplot(224)
    plt.cla() #n_fft = 1024 , hop_length = 512/ 320
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=40960, hop_length=512, n_mels=128, power=2.0)#,
    img = librosa.display.specshow(librosa.power_to_db(mel_spect, ref=np.max), x_axis='time', y_axis='mel', fmax=16400)#,cmap ="Greys")
    #img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='mel', x_axis='time', fmax=16400)#, ax=ax)
    plt.title('Mel spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Freq [Hz]')
    plt.xlim(0,10)
    #plt.gca().set_xlim(0,10) (beat_times[0]+0.5
    rect = mpatches.Rectangle((beat_times[0] ,500),4.5,2048, color= "cyan",linewidth=3, alpha = 0.2)
    plt.text(beat_times[3], 2448, 'one_cycle ',color="cyan")
    plt.gca().add_patch(rect)
    rect1 = mpatches.Rectangle((beat_times[int(len(beat_times)/2)],500),4.5,2048, color= "lime",linewidth=3, alpha = 0.2)
    plt.text(beat_times[int(len(beat_times)/2)+3], 2448, 'two_cycle',color="lime")
    plt.gca().add_patch(rect1)
    plt.scatter(beat_times, [ 1000 for _ in beat_times], color="pink", s=30)
    vv = [beat_times[i] if  i % 3==1 else ""  for i in range(len(beat_times))]
    for id, idx in enumerate(vv):
        #if not id == 0:
            plt.axvline(x = idx, color="violet", linestyle="-.")

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
    global fs,fl,fdd
    rows = 2
    cols = 2

    # Generate indices for the grid
    plt.figure(figsize=(18, 9))
    fdd = filedialog.askdirectory()
    fs = glob.glob(fdd+"/**/*.wav", recursive=True)
    fs.sort()
    fl = 0
    filepath = fs[0]

    y, sr = librosa.load(filepath)  # sr 為採樣頻率
    plot221_4(fl, fs, y, sr)
    plot_4button()
    plt.show()
