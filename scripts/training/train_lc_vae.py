"""
Usage example:

python ./scripts/training/train_lc_vae.py \
        --model-config-path=./scripts/training/config/new_models/lc_vae.json \
        --trainer-config-path=./scripts/training/config/new_trainers/lc_vae_trainer.json \
        --train-dataset-config-path=./scripts/training/config/datasets/train_dataset.json \
        --val-dataset-config-path=./scripts/training/config/datasets/val_dataset.json \
        --hierarchical-decoder \
        --gpus=0 \
        --attribute=contour
"""
import logging

from resolv_ml.training.callbacks import LearningRateLoggerCallback

import utilities

if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train latent constraint VAE model.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    args = arg_parser.parse_args()

    logging.getLogger().setLevel(args.logging_level)

    strategy = utilities.get_distributed_strategy(args.gpus, args.gpu_memory_growth)
    with strategy.scope():
        train_data, val_data, input_shape = utilities.load_datasets(
            train_dataset_config_path=args.train_dataset_config_path,
            val_dataset_config_path=args.val_dataset_config_path,
            trainer_config_path=args.trainer_config_path,
            attribute=args.attribute
        )
        lc_vae = utilities.get_lc_vae_model(
            model_config_path=args.model_config_path,
            trainer_config_path=args.trainer_config_path
        )
        lc_vae.build(input_shape)
        trainer = utilities.get_lc_vae_trainer(
            model=lc_vae, trainer_config_path=args.trainer_config_path
        )
        history = trainer.train(
            train_data=train_data[0],
            train_data_cardinality=train_data[1],
            validation_data=val_data[0],
            validation_data_cardinality=val_data[1],
            custom_callbacks=[LearningRateLoggerCallback()]
        )
