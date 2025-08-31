import logging
import os
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from resolv_mir.note_sequence.io import midi_io
from resolv_ml.models.dlvm.misc.latent_diffusion import LatentDiffusion
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities


def test_model_generation(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / attribute / Path(args.model_path).stem / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        utilities.set_log_handler(output_dir=output_dir.parent, log_level=getattr(logging, vargs.logging_level))
        model = keras.saving.load_model(args.model_path, compile=False)
        model.compile(run_eagerly=True)
        model.trainable = False
        input_sequences, input_sequences_attributes = utilities.load_flat_dataset(dataset_path=args.test_dataset_path,
                                                                                  sequence_length=args.sequence_length,
                                                                                  attribute=attribute,
                                                                                  batch_size=args.batch_size,
                                                                                  parse_sequence_feature=True)
        steps = args.dataset_cardinality // args.batch_size
        dataset_min_attr_val = args.min_attr if args.min_attr > 0 else np.min(input_sequences_attributes)
        dataset_max_attr_val = args.max_attr if args.max_attr > 0 else np.max(input_sequences_attributes)
        attr_conditioning_labels = keras.ops.expand_dims(
            keras.ops.linspace(
                start=dataset_min_attr_val, stop=dataset_max_attr_val, num=args.batch_size * steps
            ), axis=-1
        )
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Converting input sequences to MIDI and save to disk...")
        num_sequences = input_sequences.shape[0]
        representation = PitchSequenceRepresentation(args.sequence_length)
        sequences_to_save, attr_idxes = input_sequences, np.arange(num_sequences)
        if 0 < args.num_midi_to_save < num_sequences:
            attr_idxes = [random.randint(0, num_sequences - 1) for _ in range(args.num_midi_to_save)]
            sequences_to_save = keras.ops.take(input_sequences, indices=attr_idxes, axis=0)
        for idx, input_sequence in enumerate(sequences_to_save):
            input_note_sequence = representation.to_canonical_format(input_sequence.squeeze(), attributes=None)
            filename = f"midi/inputs/example_{attribute}_{input_sequences_attributes[attr_idxes[idx]][0]:5f}_{idx}.midi"
            midi_io.note_sequence_to_midi_file(
                note_sequence=input_note_sequence,
                output_file=Path(args.output_path) / attribute / Path(args.model_path).stem / filename
            )
        # -------------------------------------------------------------------------------------
        if isinstance(model, LatentDiffusion):
            logging.info("----------------------------- TEST FROZEN VAE -----------------------------")
            latent_codes = model._vae.get_latent_codes(keras.ops.convert_to_tensor(args.batch_size * steps))
            decoder_inputs = keras.ops.convert_to_tensor(args.sequence_length, dtype="int32")
            vae_generated_sequences = model._vae.decode(inputs=(latent_codes, decoder_inputs))
            plots(sequences=vae_generated_sequences.numpy(),
                  labels=keras.ops.squeeze(attr_conditioning_labels),
                  output_dir=output_dir / "vae",
                  attribute=attribute,
                  args=args)
            logging.info("----------------------------- TEST DIFFUSION CONDITIONING -----------------------------")
            num_samples = keras.ops.full((args.batch_size,), args.batch_size, dtype="int32")
            decoder_inputs = keras.ops.full((args.batch_size,), args.sequence_length, dtype="int32")
            model._diffusion._sampling_timesteps = args.sampling_steps
            diff_linspace_generated_sequences = []
            diff_dataset_generated_sequences = []
            for i in range(steps):
                logging.info(f"Generating sequences with Diffusion for batch {i}/{steps}...")
                batch_lin_attr_labels = attr_conditioning_labels[i * args.batch_size:(i + 1) * args.batch_size]
                diff_lin_seqs, _, _ = model.predict(
                    x=(num_samples, batch_lin_attr_labels, decoder_inputs), batch_size=args.batch_size
                )
                diff_linspace_generated_sequences.extend(diff_lin_seqs.tolist())
                batch_data_attr_labels = input_sequences_attributes[i * args.batch_size:(i + 1) * args.batch_size]
                diff_data_seqs, _, _ = model.predict(
                    x=(num_samples, batch_data_attr_labels, decoder_inputs), batch_size=args.batch_size
                )
                diff_dataset_generated_sequences.extend(diff_data_seqs.tolist())
            logging.info("------------------------------------ LINSPACE ------------------------------------")
            plots(sequences=np.array(diff_linspace_generated_sequences),
                  labels=keras.ops.squeeze(attr_conditioning_labels),
                  output_dir=output_dir / "diffusion" / "linspace",
                  attribute=attribute,
                  args=args)
            logging.info("------------------------------------ DATASET ------------------------------------")
            plots(sequences=np.array(diff_dataset_generated_sequences),
                  labels=keras.ops.squeeze(input_sequences_attributes),
                  output_dir=output_dir / "diffusion" / "dataset",
                  attribute=attribute,
                  args=args)
        else:
            logging.info("----------------------------- TEST AR-VAE -----------------------------")
            latent_codes = model.get_latent_codes(keras.ops.convert_to_tensor(args.batch_size * steps)).numpy()
            z_lin_conditioning_labels = attr_conditioning_labels.numpy()
            z_data_conditioning_labels = input_sequences_attributes
            # Preprocess labels if needed
            if model._attribute_processing_layer:
                z_lin_conditioning_labels = model._attribute_processing_layer(
                    z_lin_conditioning_labels, training=False
                ).numpy()
                z_data_conditioning_labels = model._attribute_processing_layer(
                    z_data_conditioning_labels, training=False
                ).numpy()
            decoder_inputs = keras.ops.convert_to_tensor(args.sequence_length, dtype="int32")
            logging.info("------------------------------------ LINSPACE ------------------------------------")
            latent_codes[:, args.regularized_dimension] = z_lin_conditioning_labels.squeeze()
            generated_lin_sequences = model.decode(inputs=(latent_codes, decoder_inputs))
            plots(sequences=generated_lin_sequences.numpy(),
                  labels=keras.ops.squeeze(attr_conditioning_labels),
                  output_dir=output_dir / "linspace",
                  attribute=attribute,
                  args=args)
            logging.info("------------------------------------ DATASET ------------------------------------")
            latent_codes[:, args.regularized_dimension] = z_data_conditioning_labels.squeeze()
            generated_data_sequences = model.decode(inputs=(latent_codes, decoder_inputs))
            plots(sequences=generated_data_sequences.numpy(),
                  labels=keras.ops.squeeze(input_sequences_attributes),
                  output_dir=output_dir / "dataset",
                  attribute=attribute,
                  args=args)


def kld(true, gen, bins='auto'):
    bins = len(np.unique(true))
    hist_true, bins = np.histogram(true, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen, bins=bins, density=True)

    # Add a small constant to avoid division by zero and log(0) errors.
    epsilon = 1e-12
    hist_true = hist_true + epsilon
    hist_gen = hist_gen + epsilon

    return scipy.stats.entropy(hist_true, hist_gen, base=2)


def plots(sequences, labels, output_dir: Path, attribute: str, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Computing sequences attributes...")
    sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
        sequences, attribute, args.sequence_length
    )
    labels = labels.numpy()
    # ------------------------------------------------------------------------------------------------------------
    logging.info("Plot generated sequences attributes histogram...")
    filename = f'{str(output_dir)}/histogram_generated_{attribute}_{args.histogram_bins[0]}_bins.png'
    logging.info(f"Plotting generated histogram with {args.histogram_bins[0]} bins for attribute {attribute}...")
    with plt.rc_context(utilities.get_matplotlib_context()):
        plt.hist(sequences_attrs, bins=args.histogram_bins[0], density=True, stacked=True, color='C0', alpha=0.7)
        plt.grid(linestyle=':')
        plt.xlabel(r'$a$')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()
    # ------------------------------------------------------------------------------------------------------------
    logging.info(f"KLD {kld(np.array(sequences_attrs), np.array(labels))}")
    # ------------------------------------------------------------------------------------------------------------
    logging.info("Computing coefficients and plotting graph with the best linear fitting model...")
    pearson_coefficient = scipy.stats.pearsonr(sequences_attrs, labels)
    spearman_coefficient = scipy.stats.spearmanr(sequences_attrs, labels)
    slope, intercept = plot_reg_dim_vs_attribute(output_path=str(output_dir / 'labels_vs_attribute.png'),
                                                 labels=labels,
                                                 attribute_data=sequences_attrs)
    logging.info(f"Best linear model fit parameters. Slope: {slope}, Intercept: {intercept}")
    logging.info(f"Pearson coefficient {pearson_coefficient.statistic}.")
    logging.info(f"Spearman coefficient {spearman_coefficient.statistic}.")
    logging.info(f"Generated {hold_note_start_seq_count} sequences that start with an hold note token "
                 f"({hold_note_start_seq_count * 100 / args.dataset_cardinality:.2f}%).")
    # ------------------------------------------------------------------------------------------------------------
    logging.info("Converting generated sequences to MIDI and save to disk...")
    num_sequences = sequences.shape[0]
    representation = PitchSequenceRepresentation(args.sequence_length)
    sequences_to_save, attr_idxes = sequences, np.arange(num_sequences)
    if 0 < args.num_midi_to_save < num_sequences:
        attr_idxes = [random.randint(0, num_sequences - 1) for _ in range(args.num_midi_to_save)]
        sequences_to_save = keras.ops.take(sequences, indices=attr_idxes, axis=0)
    for idx, generated_sequence in enumerate(sequences_to_save):
        generated_note_sequence = representation.to_canonical_format(generated_sequence, attributes=None)
        filename = (f"midi/gen/{output_dir.stem}/gen_{attribute}_cond_{labels[attr_idxes[idx]]:5f}"
                    f"_seq_{sequences_attrs[attr_idxes[idx]]:5f}_{idx}.midi")
        midi_io.note_sequence_to_midi_file(
            note_sequence=generated_note_sequence,
            output_file=Path(args.output_path) / attribute / Path(args.model_path).stem / filename
        )


def plot_reg_dim_vs_attribute(output_path: str,
                              labels,
                              attribute_data):
    slope, intercept = np.polyfit(labels, attribute_data, 1)
    with plt.rc_context(utilities.get_matplotlib_context()):
        plt.scatter(labels, attribute_data, color='C0', s=5, alpha=0.35, edgecolors='none')
        plt.plot(labels, slope * labels + intercept, color='#fd5656')
        plt.xlabel(r'$a_l$')
        plt.ylabel(r'$a_g$')
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
    return slope, intercept


if __name__ == '__main__':
    parser = utilities.get_arg_parser(description="")
    parser.add_argument('--model-path', required=True, help='Path to the .keras model checkpoint.')
    parser.add_argument('--dataset-cardinality', help='Cardinality of the test dataset.', required=True, type=int)
    parser.add_argument('--sampling-steps', help='Number of steps for diffusion sampling.', default=100, required=False,
                        type=int)
    parser.add_argument('--num-midi-to-save', help='Number of generated sequences to save as MIDI file. The N '
                                                   'sequences will be chosen randomly.', default=-1, required=False,
                        type=int)
    parser.add_argument('--min-attr', help='', required=False, default=-1.0, type=float)
    parser.add_argument('--max-attr', help='', required=False, default=-1.0, type=float)
    parser.add_argument('--regularized-dimension', help='Index of the latent code regularized dimension.',
                        required=False, type=int, default=0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    test_model_generation(vargs)
