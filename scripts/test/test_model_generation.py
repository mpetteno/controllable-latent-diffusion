import logging
import os
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from resolv_mir.note_sequence.io import midi_io
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities


def test_model_generation(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / attribute / Path(args.model_path).stem / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         parse_sequence_feature=True)
        model = keras.saving.load_model(args.model_path, compile=False)
        model.compile(run_eagerly=True)
        model.trainable = False
        model._diffusion._sampling_timesteps = args.sampling_steps
        # Generate latent codes and sequences with VAE
        vae_generated_sequences, vae_latent_codes, input_sequences, input_sequences_attributes, _ = model._vae.predict(
            dataset, steps=args.dataset_cardinality//args.batch_size
        )
        plots(sequences=vae_generated_sequences,
              latent_codes=vae_latent_codes,
              output_dir=output_dir / "vae",
              attribute=attribute,
              args=args)
        # Generate latent codes and sequences with Diffusion
        denoised_latent_codes = []
        diff_generated_sequences = []
        dataset_min_attr_val = np.min(input_sequences_attributes)
        dataset_max_attr_val = np.max(input_sequences_attributes)
        reg_dim_values = keras.ops.linspace(
            start=dataset_min_attr_val, stop=dataset_max_attr_val, num=args.dataset_cardinality
        )
        for i in range(args.dataset_cardinality // args.batch_size):
            logging.info(f"Generating sequences with Diffusion for batch {i}/"
                         f"{args.dataset_cardinality // args.batch_size}...")
            latent_codes = model.get_latent_codes(keras.ops.convert_to_tensor(args.batch_size)).numpy()
            if args.control_reg_dim:
                latent_codes[:, args.regularized_dimension] = reg_dim_values[i*args.batch_size:(i+1)*args.batch_size]
            diff_seqs, _, diff_latent_codes = model.sample(
                inputs=(latent_codes, keras.ops.convert_to_tensor(args.sequence_length))
            )
            denoised_latent_codes.extend(diff_latent_codes.numpy()[:, -1, :].tolist())
            diff_generated_sequences.extend(diff_seqs.numpy().tolist())
        plots(sequences=np.array(diff_generated_sequences),
              latent_codes=np.array(denoised_latent_codes),
              output_dir=output_dir / "diffusion",
              attribute=attribute,
              args=args)


def plots(sequences, latent_codes, output_dir: Path, attribute: str, args, sequences_attrs=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    # compute sequences attributes
    if not sequences_attrs:
        sequences_attrs, _ = utilities.compute_sequences_attributes(sequences, attribute, args.sequence_length)
    # plot generated sequences attributes histogram
    filename = f'{str(output_dir)}/histogram_generated_{attribute}_{args.histogram_bins}_bins.png'
    logging.info(f"Plotting generated histogram with {args.histogram_bins} bins for attribute {attribute}...")
    plt.hist(sequences_attrs, bins=args.histogram_bins, density=True, stacked=True, color='blue', alpha=0.7)
    plt.grid(linestyle=':')
    plt.savefig(filename, format='png', dpi=300)
    plt.close()
    # compute Pearson coefficient and plot graph with the best linear fitting model
    reg_dim_data = latent_codes[:, args.regularized_dimension]
    correlation_matrix = np.corrcoef(reg_dim_data, sequences_attrs)
    pearson_coefficient = correlation_matrix[0, 1]
    slope, intercept = plot_reg_dim_vs_attribute(output_path=str(output_dir / 'reg_dim_vs_attribute.png'),
                                                 reg_dim_data=reg_dim_data,
                                                 reg_dim_idx=args.regularized_dimension,
                                                 attribute_data=sequences_attrs)
    # convert generated sequences to MIDI and save to disk
    representation = PitchSequenceRepresentation(args.sequence_length)
    seq_to_save_count = min(args.dataset_cardinality, args.num_midi_to_save)
    random_idxes = [random.randint(0, args.dataset_cardinality - 1) for _ in range(seq_to_save_count)]
    for idx, generated_sequence in enumerate(keras.ops.take(sequences, indices=random_idxes, axis=0)):
        generated_note_sequence = representation.to_canonical_format(generated_sequence, attributes=None)
        filename = (f"midi/{output_dir.stem}/{attribute}_"
                    f"{latent_codes[random_idxes[idx], args.regularized_dimension]:.2f}.midi")
        midi_io.note_sequence_to_midi_file(
            note_sequence=generated_note_sequence,
            output_file=Path(args.output_path) / attribute / Path(args.model_path).stem / filename
        )
    logging.info(f"Best linear model fit parameters. Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    logging.info(f"Pearson coefficient {pearson_coefficient:.2f}.")


def plot_reg_dim_vs_attribute(output_path: str,
                              reg_dim_data,
                              reg_dim_idx,
                              attribute_data):
    slope, intercept = np.polyfit(reg_dim_data, attribute_data, 1)
    plt.scatter(reg_dim_data, attribute_data, color='blue', s=5)
    plt.plot(reg_dim_data, slope * reg_dim_data + intercept, color='red')
    plt.xlabel(f'$z_{reg_dim_idx}$')
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
    return slope, intercept


if __name__ == '__main__':
    parser = utilities.get_arg_parser("")
    parser.add_argument('--control-reg-dim', action="store_true",
                        help='Control the regularized latent dimension of the sampled latent codes using the min and '
                             'max values provided in `--latent-min-val` and `--latent-max-val.`')
    parser.add_argument('--sampling-steps', help='Number of steps for diffusion sampling.', default=100, required=False,
                        type=int)
    parser.add_argument('--num-midi-to-save', help='Number of generated sequences to save as MIDI file. '
                                                   'The N sequences will be chosen randomly.', required=True, type=int)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.', default=120, required=False,
                        type=int)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()

    # Logger settings
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO
    logger = logging.getLogger()
    logger.setLevel(vargs.logging_level)
    handler = logging.StreamHandler()
    handler.addFilter(InfoFilter())  # Apply the filter to allow only INFO messages
    logger.addHandler(handler)

    test_model_generation(vargs)
