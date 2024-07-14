import argparse
from src.evaluate import evaluate
from src.convert import convert
from src.plotbase import plotbase
from src.make_model_curve import plotmodel
def parameter():
    parser = argparse.ArgumentParser(description='Evaluate the performance of the abonomaly detection algorithm. It also supports converting the data files to a format that can be used by this program.')
    parser.add_argument('mode', choices=['evaluate', 'convert', 'plotbase', 'plotmodel'], help='Mode of the program')
    parser.add_argument('-i', '--input_dir', help='Input directory', default='data')
    parser.add_argument('-o', '--output_dir', help='Output directory', default='results')
    parser.add_argument('-t', '--theta', help='Threshold value', type=float, default=35.0)
    parser.add_argument('-c', '--threshold_cnt', help='Threshold count', type=int, default=4)
    parser.add_argument('-n', '--ncpu', help='Number of CPUs to use. Set to -1 to use half of your cores', type=int, default=-1)
    parser.add_argument('-s', '--smooth_window_Hz', help='Smooth window size in Hz', type=float, default=5.0)
    parser.add_argument('-q', '--quite', help='Disable drawing plots for each predictions', action='store_true')
    parser.add_argument('-f', '--force', help='Force to overwrite the output directory', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot listing frequency fiqure', action='store_true')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parameter()
    
    if args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'convert':
        convert(args)
    elif args.mode == 'plotbase':
        plotbase(args)
    elif args.mode == 'plotmodel':
        plotmodel(args)
    else:
        raise ValueError('Invalid mode. Choose either evaluate or convert.')