import logging
import os
import sys
from pathlib import Path
from typing import List

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import utilities


def test_distributions(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / attribute
        output_dir.mkdir(parents=True, exist_ok=True)
        set_log_handler(output_dir=output_dir, log_level=getattr(logging, vargs.logging_level))
        input_sequences_attributes = utilities.load_flat_dataset(dataset_path=args.test_dataset_path,
                                                                 sequence_length=args.sequence_length,
                                                                 attribute=attribute,
                                                                 batch_size=args.batch_size,
                                                                 parse_sequence_feature=False)
        logging.info(f"5% Quantile {np.quantile(input_sequences_attributes, 0.05, axis=0)}")
        logging.info(f"2.5% Quantile {np.quantile(input_sequences_attributes, 0.025, axis=0)}")
        logging.info(f"1% Quantile {np.quantile(input_sequences_attributes, 0.01, axis=0)}")
        logging.info(f"95% Quantile {np.quantile(input_sequences_attributes, 0.95, axis=0)}")
        logging.info(f"97.5% Quantile {np.quantile(input_sequences_attributes, 0.975, axis=0)}")
        logging.info(f"99% Quantile {np.quantile(input_sequences_attributes, 0.99, axis=0)}")
        pt_sequences_attributes = scipy.stats.boxcox(input_sequences_attributes + args.shift, args.power)
        plot_distributions(
            pt_data=pt_sequences_attributes,
            original_data=input_sequences_attributes,
            output_path=output_dir,
            power=args.power,
            shift=args.shift,
            attribute=attribute,
            x_lim=args.x_lim,
            histogram_bins=args.histogram_bins
        )


@mpl.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.size': 20, 'font.serif': 'Computer Modern Roman',
                 'lines.linewidth': 1.5})
def plot_distributions(pt_data,
                       original_data,
                       output_path: Path,
                       power: float,
                       shift: float,
                       attribute: str,
                       x_lim: float,
                       histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    for n_bins in histogram_bins:
        filename = (f'{str(histograms_output_path)}/histogram_{attribute}_power_{power:.2f}_shift_{shift:.3f}'
                    f'_bins_{n_bins}.png')
        # Create subplots
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5, 5))
        # Original distribution histogram
        counts, bins = np.histogram(original_data, bins=n_bins)
        weights = (counts / np.max(counts)) * 0.45
        axes[0].hist(bins[:-1], bins=n_bins, weights=weights, color='C1')
        axes[0].set_xlabel('$a$')
        axes[0].set_axisbelow(True)
        axes[0].yaxis.grid(linestyle=':')
        if x_lim >= 0:
            axes[0].set_xlim(right=x_lim)
        # Power transform histogram
        counts, bins = np.histogram(pt_data, bins=n_bins)
        weights = (counts / np.max(counts)) * 0.45
        axes[1].hist(bins[:-1], bins=n_bins, weights=weights, color='C0')
        axes[1].set_xlabel('$T_\lambda(a)$')
        axes[1].set_axisbelow(True)
        axes[1].yaxis.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


def set_log_handler(output_dir, log_level):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Create file handler
    file_handler = logging.FileHandler(output_dir / 'log.txt')
    file_handler.setLevel(log_level)
    # Create console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    # Use the default logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


if __name__ == '__main__':
    parser = utilities.get_arg_parser(description="")
    parser.add_argument('--power', help='Initial value for the power transform\'s power parameter.',
                        default=1.0, type=float)
    parser.add_argument('--shift', help='Initial value for the power transform\'s shift parameter.',
                        default=0.0, type=float)
    parser.add_argument('--x-lim', help='X axis limit value for original distribution histogram.',
                        default=-1, required=False, type=float)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    test_distributions(vargs)
