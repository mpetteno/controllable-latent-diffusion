import argparse
import functools
import json
import logging
import math
from pathlib import Path
from typing import List, Dict

import keras
import tensorflow as tf
from resolv_ml.models.dlvm.diffusion.ddim import DDIM
from resolv_ml.models.dlvm.misc.latent_diffusion import LatentDiffusion
from resolv_ml.models.dlvm.vae.ar_vae import AttributeRegularizedVAE
from resolv_ml.models.dlvm.vae.base import VAE
from resolv_ml.models.dlvm.vae.vanilla_vae import StandardVAE
from resolv_ml.models.nn.denoisers import DenseDenoiser
from resolv_ml.models.nn.embeddings import SinusoidalPositionalEncoding
from resolv_ml.models.seq2seq.rnn.decoders import HierarchicalRNNDecoder, RNNAutoregressiveDecoder
from resolv_ml.models.seq2seq.rnn.encoders import BidirectionalRNNEncoder
from resolv_ml.training.trainer import Trainer
from resolv_ml.utilities.regularizers.attribute import AttributeRegularizer
from resolv_ml.utilities.schedulers import get_scheduler
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation


def check_tf_gpu_availability():
    logging.info(f'Tensorflow version: {tf.__version__}')
    gpu_list = tf.config.list_physical_devices('GPU')
    if len(gpu_list) > 0:
        logging.info(f'Num GPUs Available: {len(gpu_list)}. List: {gpu_list}')
    return gpu_list


def set_visible_devices(gpu_ids: List[int] = None, memory_growth: bool = False):
    gpu_list = check_tf_gpu_availability()

    if not gpu_list:
        raise SystemExit("GPU not available.")

    if gpu_ids:
        selected_gpus = [gpu_list[gpu] for gpu in gpu_ids]
        logging.info(f"Using provided GPU device {selected_gpus}.")
    else:
        logging.info(f"No GPU ids provided. Using default GPU device {gpu_list[0]}.")
        selected_gpus = [gpu_list[0]]

    tf.config.set_visible_devices(selected_gpus, 'GPU')

    for gpu in selected_gpus:
        logging.info(f"Setting GPU device {gpu} memory growth to {memory_growth}.")
        tf.config.experimental.set_memory_growth(gpu, memory_growth)

    return selected_gpus


def get_distributed_strategy(gpu_ids: List[int] = None, memory_growth: bool = False) -> tf.distribute.Strategy:
    selected_gpus = set_visible_devices(gpu_ids, memory_growth)
    selected_gpus_name = [selected_gpu.name.replace("/physical_device:", "") for selected_gpu in selected_gpus]
    if len(selected_gpus) > 1:
        logging.info(f"Using MirroredStrategy on selected GPUs: {selected_gpus_name}")
        strategy = tf.distribute.MirroredStrategy(devices=selected_gpus_name)
    else:
        logging.info(f"Using OneDeviceStrategy on selected GPU: {selected_gpus_name[0]}")
        strategy = tf.distribute.OneDeviceStrategy(device=selected_gpus_name[0])
    return strategy


def get_vae_model(model_config_path: str,
                  trainer_config_path: str,
                  hierarchical_decoder: bool = False,
                  attribute_proc_layer: keras.Layer = None,
                  attribute_regularizers: Dict[str, AttributeRegularizer] = None,
                  inference_layer: keras.Layer = None) -> VAE:
    with open(model_config_path) as file:
        model_config = json.load(file)
        schedulers_config = model_config["schedulers"]

    with open(trainer_config_path) as file:
        fit_config = json.load(file)["fit"]

    encoder_config = model_config["encoder"]
    encoder = BidirectionalRNNEncoder(
        enc_rnn_sizes=encoder_config["enc_rnn_sizes"],
        embedding_layer=keras.layers.Embedding(
            input_dim=model_config["num_classes"],
            output_dim=model_config["embedding_size"],
            name="encoder_embedding"
        ),
        dropout=encoder_config["dropout"]
    )

    if hierarchical_decoder:
        hier_decoder_config = model_config["hier_decoder"]
        core_decoder_config = hier_decoder_config["core_decoder"]
        decoder = HierarchicalRNNDecoder(
            level_lengths=hier_decoder_config["level_lengths"],
            dec_rnn_sizes=hier_decoder_config["dec_rnn_sizes"],
            dropout=hier_decoder_config["dropout"],
            sampling_scheduler=get_scheduler(
                schedule_type=schedulers_config["sampling_probability"]["type"],
                schedule_config=schedulers_config["sampling_probability"]["config"]
            ),
            core_decoder=RNNAutoregressiveDecoder(
                dec_rnn_sizes=core_decoder_config["dec_rnn_sizes"],
                num_classes=model_config["num_classes"],
                embedding_layer=keras.layers.Embedding(
                    input_dim=model_config["num_classes"],
                    output_dim=model_config["embedding_size"],
                    name="decoder_embedding"
                ),
                dropout=core_decoder_config["dropout"]
            )
        )
    else:
        decoder_config = model_config["decoder"]
        decoder = RNNAutoregressiveDecoder(
            dec_rnn_sizes=decoder_config["dec_rnn_sizes"],
            num_classes=model_config["num_classes"],
            embedding_layer=keras.layers.Embedding(
                input_dim=model_config["num_classes"],
                output_dim=model_config["embedding_size"],
                name="decoder_embedding"
            ),
            dropout=decoder_config["dropout"],
            sampling_scheduler=get_scheduler(
                schedule_type=schedulers_config["sampling_probability"]["type"],
                schedule_config={
                    **schedulers_config["sampling_probability"]["config"],
                    "total_steps": fit_config["total_steps"]
                }
            )
        )

    if attribute_regularizers:
        return AttributeRegularizedVAE(
            z_size=model_config["z_size"],
            input_processing_layer=encoder,
            generative_layer=decoder,
            attribute_processing_layer=attribute_proc_layer,
            attribute_regularizers=attribute_regularizers,
            inference_layer=inference_layer,
            free_bits=model_config["free_bits"],
            div_beta_scheduler=get_scheduler(
                schedule_type=schedulers_config["kl_div_beta"]["type"],
                schedule_config={
                    **schedulers_config["kl_div_beta"]["config"],
                    "total_steps": fit_config["total_steps"]
                }
            )
        )
    else:
        return StandardVAE(
            z_size=model_config["z_size"],
            input_processing_layer=encoder,
            generative_layer=decoder,
            free_bits=model_config["free_bits"],
            div_beta_scheduler=get_scheduler(
                schedule_type=schedulers_config["kl_div_beta"]["type"],
                schedule_config={
                    **schedulers_config["kl_div_beta"]["config"],
                    "total_steps": fit_config["total_steps"]
                }
            )
        )


def get_latent_diffusion_model(model_config_path: str) -> LatentDiffusion:
    with open(model_config_path) as file:
        latent_diffusion_config = json.load(file)

    with open(latent_diffusion_config["vae_config_path"]) as file:
        vae_config = json.load(file)

    with open(latent_diffusion_config["diffusion_config_path"]) as file:
        diffusion_config = json.load(file)
        denoiser_config = diffusion_config["dense_denoiser"]
        pos_emb_config = denoiser_config["positional_embedding"]

    diffusion_model = DDIM(
        z_shape=(vae_config["z_size"],),
        denoiser=DenseDenoiser(
            units=denoiser_config["units"],
            num_layers=denoiser_config["num_layers"],
            positional_encoding_layer=SinusoidalPositionalEncoding(
                embedding_dim=pos_emb_config["embedding_dim"],
                max_seq_length=pos_emb_config["max_seq_length"],
                frequency_base=pos_emb_config["frequency_base"],
                frequency_scaling=pos_emb_config["frequency_scaling"],
                lookup_table=pos_emb_config["lookup_table"]
            )
        ),
        timesteps=diffusion_config["timesteps"],
        sampling_timesteps=diffusion_config["sampling_timesteps"],
        timesteps_scheduler_type=diffusion_config["timesteps_scheduler_type"],
        noise_level_conditioning=diffusion_config["noise_level_conditioning"],
        loss_fn=diffusion_config["loss_fn"],
        eta=diffusion_config["eta"],
        noise_schedule_type=diffusion_config["noise_scheduler"]["type"],
        noise_schedule_start=diffusion_config["noise_scheduler"]["start"],
        noise_schedule_end=diffusion_config["noise_scheduler"]["end"]
    )
    return LatentDiffusion(vae=latent_diffusion_config["vae_weights_path"], diffusion=diffusion_model)


def load_datasets(train_dataset_config_path: str,
                  val_dataset_config_path: str,
                  trainer_config_path: str,
                  attribute: str = None):
    train_data, input_shape, train_length = load_pitch_seq_dataset(dataset_config_path=train_dataset_config_path,
                                                                   trainer_config_path=trainer_config_path,
                                                                   attribute=attribute,
                                                                   training=True)
    val_data, _, val_length = load_pitch_seq_dataset(dataset_config_path=val_dataset_config_path,
                                                     trainer_config_path=trainer_config_path,
                                                     attribute=attribute,
                                                     training=False)
    return (train_data, train_length), (val_data, val_length), input_shape


def load_pitch_seq_dataset(dataset_config_path: str,
                           trainer_config_path: str,
                           attribute: str = None,
                           training: bool = False) -> tf.data.TFRecordDataset:
    def get_input_shape():
        input_seq_shape = batch_size, sequence_length, sequence_features
        aux_input_shape = (batch_size, 1)
        return input_seq_shape, aux_input_shape

    def map_fn(ctx, seq):
        input_seq = tf.transpose(seq["pitch_seq"])
        attributes = tf.expand_dims(ctx[attribute], axis=-1) if attribute \
            else tf.zeros(shape=(batch_size, 1))
        target = input_seq
        return (input_seq, attributes), target

    with open(dataset_config_path) as file:
        dataset_config = json.load(file)

    with open(trainer_config_path) as file:
        fit_config = json.load(file)["fit"]

    dataset_cardinality = dataset_config.pop("dataset_cardinality")
    batch_size = fit_config["batch_size"]
    total_steps = fit_config["total_steps"]
    if training:
        dataset_steps = dataset_cardinality // batch_size
        dataset_repeat_count = total_steps / dataset_steps
        dataset_repeat_count = None if dataset_repeat_count < 1 else math.ceil(dataset_repeat_count)
    else:
        dataset_repeat_count = total_steps // (fit_config['steps_per_epoch'] * fit_config['validation_freq'])

    sequence_length = dataset_config.pop("sequence_length")
    sequence_features = dataset_config.pop("sequence_features")
    representation = PitchSequenceRepresentation(sequence_length=sequence_length)
    tfrecord_loader = TFRecordLoader(
        file_pattern=dataset_config.pop("dataset_path"),
        parse_fn=functools.partial(
            representation.parse_example,
            parse_sequence_feature=True,
            attributes_to_parse=[attribute] if attribute else None
        ),
        map_fn=map_fn,
        batch_size=batch_size,
        repeat_count=dataset_repeat_count,
        shuffle_buffer_size=dataset_config.pop("shuffle_buffer_size") or dataset_cardinality,
        **dataset_config
    )
    return tfrecord_loader.load_dataset(), get_input_shape(), dataset_cardinality


def get_latent_diffusion_trainer(trainer_config_path: Path, model: keras.Model) -> Trainer:
    trainer = Trainer(model, config_file_path=trainer_config_path)
    trainer.compile(
        lr_schedule=keras.optimizers.schedules.ExponentialDecay(
            **trainer.config["compile"]["optimizer"]["config"]["learning_rate"]
        )
    )
    return trainer


def get_vae_trainer(trainer_config_path: Path, model: keras.Model) -> Trainer:
    trainer = Trainer(model, config_file_path=trainer_config_path)
    trainer.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
                 keras.metrics.SparseCategoricalAccuracy(),
                 keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="sparse_top_3_categorical_accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="sparse_top_5_categorical_accuracy")],
        lr_schedule=keras.optimizers.schedules.ExponentialDecay(
            **trainer.config["compile"]["optimizer"]["config"]["learning_rate"]
        )
    )
    return trainer


def get_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-config-path', help='Path to the diffusion model\'s configuration file.',
                        required=True)
    parser.add_argument('--hierarchical-decoder', help='Use a hierarchical decoder.', action="store_true")
    parser.add_argument('--trainer-config-path', help='Path to the trainer\'s configuration file.', required=True)
    parser.add_argument('--train-dataset-config-path', help='Path to the train dataset\'s configuration file.',
                        required=True)
    parser.add_argument('--val-dataset-config-path', help='Path to the validation dataset\'s configuration file.',
                        required=True)
    parser.add_argument('--gpus', nargs="+", help='ID of GPUs to use for training.', required=False,
                        default=[], type=int)
    parser.add_argument('--gpu-memory-growth', help='Set memory growth for the selected GPUs.', action="store_true")
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    return parser
