import os
import pandas as pd
from tqdm import tqdm
from .utils import list_files_and_directories
from time import time
from multiprocessing import Pool, cpu_count
from colorama import Fore
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def plot_diagram(pattern_df_smooth, savdir, savname):
    """plot the normal pattern

    Args:
        pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
        savdir (str): directory to save the plot
    """
    import os
    # Define the grid size
    rows = 4
    cols = 3

    # Generate indices for the grid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.stack((y, x), axis=-1).reshape(-1, 2)

    _, axs = plt.subplots(4, 3, figsize=(30, 20), constrained_layout=True)
    header = ["Green_X", "Blue_X", "Red_X", "Purple_X", "Yellow_X", "Orange_X", "Green2_X", "Blue2_X", "Red2_X", "Pink2_X", "Yellow2_X", "Orange2_X"]
    colorful = ["green", "blue", "red", "purple", "yellow", "orange", "green", "blue", "red", "purple", "yellow", "orange"]


    xmin = 0
    xmax = 80
    ymin = 0
    ymax = 200


    tick_size = 24
    xaxs = pattern_df_smooth["Frequency (Hz)"]
    # yaxs = pattern_df_smooth[header]
    modtype = {}
    for i in range(12):
        ax = axs[coords[i][0], coords[i][1]]
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        peaks, width = find_peaks((pattern_df_smooth[header[i]]), height=100, width=1)

        ax.plot(xaxs, pattern_df_smooth[header[i]], color=colorful[i])
        ax.scatter(xaxs[peaks], width['peak_heights'], color="magenta", s=160)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if i < 6:
            modtype[colorful[i]] = [list(xaxs[peaks]), width['peak_heights']]
        else:
            modtype[colorful[i] + "2"] = [list(xaxs[peaks]), width['peak_heights']]

    plt.savefig(os.path.join(savdir, savname+ ".png"))
    # plt.show()
    plt.close()
    return modtype


def plot_model(mod_dic, savdir, savname):
    """
    plot 1/2/3 model
    """
    #     print(mod_dic)
    _, axs = plt.subplots(2, 1, figsize=(30, 20), constrained_layout=True)
    #     colorful = [ "red", "blue","green",  "orange","yellow","purple","red", "blue","green",  "orange","yellow","purple"]
    xmin = 0
    xmax = 13
    ymin = 0
    ymax = 400
    tick_size = 24
    colorful2 = ["green", "blue", "red", "purple", "yellow", "orange", "green2", "blue2", "red2", "purple2", "yellow2", "orange2"]
    header2 = ["Red", "Blue", "Green", "Orange", "Yellow", "Purple", "Red2", "Blue2", "Green2", "Orange2", "Yellow2", "Purple2"]

    global modvalue1, modvalue2
    modvalue1 = []
    modvalue2 = []
    for ii in range(1, 4):
        for vv in range(3, -1, -1):
            modvalue1.append(list(mod_dic.values())[vv * ii - 1][1][0])
    for ii in range(1, 4):
        for vv in range(3, -1, -1):
            modvalue2.append(list(mod_dic.values())[vv * ii - 1][1][1])
    modval = [modvalue1, modvalue2]

    # sort dict from old colorful2 sequence to new header2 sequence
    try:
        for id in range(12):
            mod_dic [ header2[id] ]  = mod_dic[ colorful2[id] ]
            del mod_dic[ colorful2[id] ]
    except:
        print("Err!!")

    for idx in range(2):
        ax = axs[idx]
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.set_title('model_' + str(idx + 1), size=32)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(list(mod_dic.keys()), modval[idx])
        ax.scatter(list(mod_dic.keys()), modval[idx], color="cyan", s=160)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    plt.savefig(os.path.join(savdir, savname + "_model.png"))
    # plt.show()
    plt.close()

def plotmodel(args):
    """Read already converted spectrum format to form/make model curve .

    Args:
        args (argparse.Namespace): CLI arguments
    """
    start_time = time()

    # Create output directory and neccesary subdirectories if they does not exist
    if (os.path.exists(args.output_dir) == False):
        os.makedirs(args.output_dir)
    else:
        if args.force == 1:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            print(Fore.RED + f"Output directory {args.output_dir} already exists. Please remove it or use -f option to overwrite.")
            return
    files, dirs = list_files_and_directories(args.input_dir)
    for d in dirs:
        if (os.path.exists(os.path.join(args.output_dir, d)) == False):
            os.makedirs(os.path.join(args.output_dir, d))
    df = pd.read_csv("./dataset400/pattern/2024-06-26_448p-5_Spec.csv")
    modtype = plot_diagram(df, args.output_dir , "1")
    plot_model(modtype, args.output_dir , "1")
    print(f"Conversion completed in {time() - start_time:.2f} seconds.")