"""
Usage example:

    python ./scripts/ml/training/train_latent_diffusion.py \
        --model-config-path=./scripts/training/config/local/models/latent_diffusion.json \
        --trainer-config-path=./scripts/training/config/local/trainers/latent_diffusion_trainer.json \
        --train-dataset-config-path=./scripts/training/config/local/datasets/train_dataset.json \
        --val-dataset-config-path=./scripts/training/config/local/datasets/val_dataset.json \
        --attribute=contour \
        --gpus=0

"""
import logging

from resolv_ml.training.callbacks import LearningRateLoggerCallback

from scripts.training import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train latent diffusion model.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    arg_parser.add_argument('--power', help='BoxCox power parameter.', required=False, default=1.0, type=float)
    arg_parser.add_argument('--shift', help='BoxCox shift parameter.', required=False, default=0.0, type=float)
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
        latent_diffusion = utilities.get_latent_diffusion_model(
            model_config_path=args.model_config_path,
            power=args.power,
            shift=args.shift if args.shift else 1e-5
        )
        latent_diffusion.build(input_shape)
        trainer = utilities.get_latent_diffusion_trainer(
            model=latent_diffusion, trainer_config_path=args.trainer_config_path
        )
        history = trainer.train(
            train_data=train_data[0],
            train_data_cardinality=train_data[1],
            validation_data=val_data[0],
            validation_data_cardinality=val_data[1],
            custom_callbacks=[LearningRateLoggerCallback()]
        )
