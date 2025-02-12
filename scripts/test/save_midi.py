import argparse
import logging
from pathlib import Path

from resolv_mir.note_sequence.io import midi_io
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from scripts.test import utilities


def save_midi(args):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = utilities.load_flat_dataset(dataset_path=args.dataset_path,
                                          sequence_length=args.sequence_length,
                                          attribute='contour',
                                          batch_size=args.batch_size,
                                          parse_sequence_feature=True)
    representation = PitchSequenceRepresentation(args.sequence_length)
    for idx, sequence in enumerate(dataset):
        logging.info(f"Processing sequence {idx} / {len(dataset)}...")
        sequence = sequence.squeeze()
        note_sequence = representation.to_canonical_format(sequence, attributes=None)
        midi_io.note_sequence_to_midi_file(
            note_sequence=note_sequence,
            output_file=output_path / f"example_{idx}.midi"
        )
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='Path to the dataset containing MIDI representations.', required=True)
    parser.add_argument('--sequence-length', help='Length of the sequences in the dataset.', required=True, type=int)
    parser.add_argument('--output-path', help='Path where MIDI files will be saved.', required=True)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    vargs = parser.parse_args()
    save_midi(vargs)
