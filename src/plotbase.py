import pandas as pd
import os
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
from time import time
from colorama import Fore
import shutil
from .utils import get_freq_delta
from scipy.signal import find_peaks
from random import randrange

header = ["Green_X", "Blue_X", "Red_X", "Purple_X", "Yellow_X", "Orange_X", "Green2_X", "Blue2_X", "Red2_X", "Pink2_X", "Yellow2_X", "Orange2_X"]

colorful = ["red", "blue", "pink", "orange", "yellow", "pink", "purple","red", "blue", "pink", "orange", "yellow", "pink", "purple","red", "blue", "pink", "orange", "yellow", "pink"]
# Define the grid size
rows = 4
cols = 3

# Generate indices for the grid
x, y = np.meshgrid(np.arange(cols), np.arange(rows))
coords = np.stack((y, x), axis=-1).reshape(-1, 2)

pattern_subdir = 'pattern'
normal_subdir  = 'normal'
abnormal_subdir = 'abnormal'

def find_plot_peak():

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    df5 = pd.read_csv("Meter_time_domain_csv.csv")
    # signal = df["Amplitude"].values
    # N = len(signal)
    # sampling_rate = 25600
    # yf = fft(signal, norm='forward')
    # xf = fftfreq(N, 1 / sampling_rate)[:N // 2]
    # plt.figure(figsize=(10, 6))
    # peaks4, width = find_peaks(np.abs(yf[:N // 2]), height=0.1, width=10)
    #
    # plt.plot(xf, np.abs(yf[:N // 2]), "-", xf[peaks4], width['peak_heights'], "x")
    xf5 = df5["Frequency(Hz)"]
    yf5 = df5["dB"]

    plt.figure(figsize=(10, 6))

    peaks5, width5 = find_peaks((yf5), height=10, width=1)

    plt.plot(xf5, (yf5), "-", xf5[peaks5], width5['peak_heights'], "x")

    plt.gca().set_xlim(0, 80)
    plt.show()
    # print(type(width5['peak_heights'][0]), width5['peak_heights'][0])

#==================================================================================================
def plot_base_pattern_collection(pattern_df_smooth, savdir, fn, windows_lens):
    """plot the normal pattern

    Args:
        pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
        savdir (str): directory to save the plot
    """
    global colorful
    _, axs = plt.subplots(4, 3, figsize=(30, 20), constrained_layout=True)
    ymin = 0
    ymax = 150

    colorA = "black"
    alphaA = 1

    tick_size = 32
    # xaxs = pattern_df_smooth["Frequency (Hz)"][:-4]
    # df = pattern_df_smooth
    df_smooth = [[] for _ in range(len(pattern_df_smooth))]
    for i in range(len(pattern_df_smooth)):
        df_smooth[i] = pattern_df_smooth[i].rolling(window=windows_lens, center=True).mean().dropna()
        #df_smooth = pattern_df_smooth.rolling(window=50, center=True).mean().dropna()
    #xaxs = pattern_df_smooth[0]["Frequency (Hz)"][:-8]
    xaxs  = df_smooth[0]["Frequency (Hz)"]
    for i in range(12):
        ax = axs[coords[i][0], coords[i][1]]
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for color_index in range(len(pattern_df_smooth)):
            ax.plot(xaxs, df_smooth[color_index][header[i]], "-" ,color=colorful[color_index] )
        # ax.plot(xaxs, df_smooth[1][header[i]], alpha=alphaA, color="blue")
        # ax.plot(xaxs, df_smooth[2][header[i]], alpha=alphaA, color="green")
        # ax.plot(xaxs, df_smooth[3][header[i]], alpha=alphaA, color="orange")
        # ax.plot(xaxs, df_smooth[4][header[i]], alpha=alphaA, color="yellow")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 300, 0.1)
        # ax.set_xticks([100, 150, 200, 250])
    plt.savefig(os.path.join(savdir, fn + ".png"))
    plt.close()
#==================================================================================================

def plot_pattern(pattern_df_smooth, savdir):
    """plot the normal pattern

    Args:
        pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
        savdir (str): directory to save the plot
    """
    _, axs = plt.subplots(4, 3, figsize=(30, 20), constrained_layout=True)
    ymin = 0
    ymax = 150

    colorA = "black"
    alphaA = 1

    tick_size = 32
    xaxs = pattern_df_smooth["Frequency (Hz)"]

    for i in range(12):
        ax = axs[coords[i][0], coords[i][1]]
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xaxs, pattern_df_smooth[header[i]], alpha=alphaA, color=colorA)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 650)
        ax.set_xticks([100, 300, 500, 650])

    plt.savefig(os.path.join(savdir, "pattern.png"))
    plt.close()


def plotbase(args):
    """plot the base diagram & model of shape diagram.

    Args:
        args (argparse.Namespace): CLI arguments

    Raises:
        ValueError: No pattern files found.
    """
    global colorful
    start_time = time()

    pattern_files = glob(os.path.join(args.input_dir, pattern_subdir, '*.csv'))
    # normal_files = glob(os.path.join(args.input_dir, normal_subdir, '*.csv'))
    # abnormal_files = glob(os.path.join(args.input_dir, abnormal_subdir, '*.csv'))

    # Check if files exist
    if len(pattern_files) == 0:
        raise ValueError('No pattern files found.')
    # if len(normal_files) == 0:
    #     raise ValueError('No normal files found.')
    # if len(abnormal_files) == 0:
    #     raise ValueError('No abnormal files found.')

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if args.force == 1:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            print(Fore.RED + f"Output directory {args.output_dir} already exists. Please remove it or use -f option to overwrite.")
            return

    # Get the frequency delta
    smooth_windows_len = int(args.smooth_window_Hz / get_freq_delta(pattern_files[0]))

    # Build the pattern
    pattern_df = None
    id = 0
    pdList = []
    for file in tqdm(pattern_files, desc="Calculate the normal pattern...", leave=True):
        df = pd.read_csv(file)[["Frequency (Hz)"] + header]
        pdList.append(df)
        id += 1
        if pattern_df is None:
            pattern_df = df
        else:
            pattern_df += df
    print("---------------< > " , len(pdList), "  : "  )
    plot_base_pattern_collection(pdList, args.output_dir, "collectionBase1",smooth_windows_len)

    pattern_df /= len(pattern_files)

    # Smooth the pattern
    pattern_df_smooth = pattern_df.rolling(window=smooth_windows_len, center=True).mean().dropna()

    # Draw the pattern
    plot_pattern(pattern_df_smooth, args.output_dir)


    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("Arguments:\n")
        f.write(f"Theta: {args.theta}\n")
        f.write(f"Smooth Window Size: {args.smooth_window_Hz}\n")
        f.write(f"Threshold Count: {args.threshold_cnt}\n")

    print('Evaluation completed in {:.3f} secs.'.format(time() - start_time))