import logging
import os
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from resolv_mir.note_sequence.io import midi_io
from resolv_ml.models.dlvm.misc.latent_diffusion import LatentDiffusion
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities


def test_model_generation(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / Path(args.model_path).stem / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        model = keras.saving.load_model(args.model_path, compile=False)
        model.compile(run_eagerly=True)
        model.trainable = False
        model._diffusion._sampling_timesteps = 1000
        # sample N = dataset_cardinality instances from model's prior
        latent_codes = model.get_latent_codes(keras.ops.convert_to_tensor(args.dataset_cardinality)).numpy()
        # control regularized dimension
        if args.control_reg_dim:
            latent_codes[:, args.regularized_dimension] = keras.ops.linspace(start=args.latent_min_val,
                                                                             stop=args.latent_max_val,
                                                                             num=args.dataset_cardinality)
        # Get normalizing flow if model uses PT regularization
        normalizing_flow_ar_layer = model._vae._regularizers.get("nf_ar", None)
        if normalizing_flow_ar_layer:
            normalizing_flow = normalizing_flow_ar_layer._normalizing_flow
            normalizing_flow._add_loss = False
        else:
            normalizing_flow = None

        # generate the sequence
        # generated_sequences= model._vae.decode(
        #   inputs=(latent_codes, keras.ops.convert_to_tensor(args.sequence_length))
        # )
        generated_sequences, _, _ = model.sample(
            inputs=(latent_codes, keras.ops.convert_to_tensor(args.sequence_length))
        )
        generated_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            generated_sequences.numpy(), attribute, args.sequence_length)
        # plot generated sequences attributes histogram
        filename = f'{str(output_dir)}/histogram_generated_{attribute}_{args.histogram_bins}_bins.png'
        logging.info(f"Plotting generated histogram with {args.histogram_bins} bins for attribute {attribute}...")
        plt.hist(generated_sequences_attrs, bins=args.histogram_bins, density=True, stacked=True, color='blue',
                 alpha=0.7)
        plt.grid(linestyle=':')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()
        # compute Pearson coefficient and plot graph with the best linear fitting model
        reg_dim_data = latent_codes[:, args.regularized_dimension]
        if normalizing_flow:
            generated_sequences_attrs = normalizing_flow(
                inputs=keras.ops.convert_to_tensor(keras.ops.expand_dims(generated_sequences_attrs, axis=-1)),
                inverse=True
            )
            generated_sequences_attrs = keras.ops.squeeze(generated_sequences_attrs)
        correlation_matrix = np.corrcoef(reg_dim_data, generated_sequences_attrs)
        pearson_coefficient = correlation_matrix[0, 1]
        slope, intercept = plot_reg_dim_vs_attribute(output_path=str(output_dir / 'reg_dim_vs_attribute.png'),
                                                     reg_dim_data=reg_dim_data,
                                                     reg_dim_idx=args.regularized_dimension,
                                                     attribute_data=generated_sequences_attrs)
        # convert generated sequences to MIDI and save to disk
        representation = PitchSequenceRepresentation(args.sequence_length)
        seq_to_save_count = min(args.dataset_cardinality, args.num_midi_to_save)
        random_idxes = [random.randint(0, args.dataset_cardinality - 1) for _ in range(seq_to_save_count)]
        for idx, generated_sequence in enumerate(keras.ops.take(generated_sequences, indices=random_idxes, axis=0)):
            generated_note_sequence = representation.to_canonical_format(generated_sequence, attributes=None)
            filename = f"midi/{attribute}_{latent_codes[random_idxes[idx], args.regularized_dimension]:.2f}.midi"
            midi_io.note_sequence_to_midi_file(generated_note_sequence,
                                               Path(args.output_path) / Path(args.model_path).stem / filename)

        # if attribute regularization is carried out by a normalizing flow, compute the minimum and maximum mapped
        # latent values
        if normalizing_flow:
            # load the test dataset and extract all the attribute values to get min and max
            dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                             sequence_length=args.sequence_length,
                                             attribute=attribute,
                                             batch_size=args.batch_size,
                                             shift=0.,
                                             parse_sequence_feature=False)
            attribute_values = np.array([x for batch in dataset for x in batch])
            attr_min_val = np.min(attribute_values)
            nf_min_val = normalizing_flow(inputs=keras.ops.convert_to_tensor([[attr_min_val]]), inverse=True).numpy()
            logging.info(f"Minimum attribute value in dataset is: {attr_min_val}. It is mapped to "
                         f"{nf_min_val[0][0]:.2f} in the latent regularized dimension.")
            attr_max_val = np.max(attribute_values)
            nf_max_val = normalizing_flow(inputs=keras.ops.convert_to_tensor([[attr_max_val]]), inverse=True).numpy()
            logging.info(f"Maximum attribute value in dataset is: {attr_max_val}. It is mapped to "
                         f"{nf_max_val[0][0]:.2f} in the latent regularized dimension.")

        logging.info(f"Best linear model fit parameters. Slope: {slope:.2f}, Intercept: {intercept:.2f}")
        logging.info(f"Pearson coefficient {pearson_coefficient:.2f}.")
        logging.info(f"Generated {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count * 100 / args.dataset_cardinality:.2f}%).")


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
    parser.add_argument('--latent-min-val', help='Minimum value for manipulation of the regularized latent dimension.',
                        default=-4.0, required=False, type=float)
    parser.add_argument('--latent-max-val', help='Maximum value for manipulation of the regularized latent dimension.',
                        default=4.0, required=False, type=float)
    parser.add_argument('--num-midi-to-save', help='Number of generated sequences to save as MIDI file. '
                                                   'The N sequences will be chosen randomly.', required=True, type=int)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.', default=120, required=False,
                        type=int)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    logging.getLogger().setLevel(vargs.logging_level)
    test_model_generation(vargs)
