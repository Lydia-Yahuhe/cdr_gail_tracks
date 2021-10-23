import numpy as np
import argparse

from featurizers import VAEFeaturizer


def build_args_parse():
    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument('--featurizer_type', help='Choose from [tdc, vae, forward_model]', default='tdc')
    args_parser.add_argument('--featurizer_save_path', help='Save path', default='.\\dataset\\checkpoint\\default.ckpt')
    args_parser.add_argument('--framerate', help='Desired FPS for the videos', type=float, default=0.25)
    args_parser.add_argument('--initial_width', help='Initial width for the videos', type=int, default=15)
    args_parser.add_argument('--initial_height', help='Initial height for the videos', type=int, default=6)
    args_parser.add_argument('--desired_width', help='Width for the videos after cropping', type=int, default=128)
    args_parser.add_argument('--desired_height', help='Height for the videos after cropping', type=int, default=128)
    args_parser.add_argument('--num_epochs', help='Number of epochs for training', type=int, default=100)
    args_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=32)
    args_parser.add_argument('--learning_rate', help='Learning rate for training', type=float, default=0.0001)
    args_parser.add_argument('--restore_model', help='featurizer_save_path', type=bool, default=False)
    return args_parser.parse_args()


def generate_dataset(width, height, split_ratio=0.8):
    dataset = []
    a_piece_length = width
    for flight_id, agent in historical_tracks.items():
        frames = np.array([generalized(track) for track in agent['tracks']], dtype=np.float64)

        length = frames.shape[0]
        size = length // a_piece_length
        end_idx = length - length % a_piece_length
        dataset += np.split(frames[:end_idx, :], size)

    dataset = np.array(dataset).reshape((-1, width, height))
    np.random.shuffle(dataset)

    split_size = int(dataset.shape[0]*split_ratio)
    train_data, test_data = dataset[:split_size], dataset[split_size:]

    print('-----------------------------------')
    print('Total size:', dataset.shape)
    print('Train size:', train_data.shape)
    print(' Test size:', test_data.shape)
    print('-----------------------------------')
    return train_data, test_data


def train(restore_model=False):
    args = build_args_parse()

    # Prepare dataset
    initial_width = args.initial_width
    initial_height = args.initial_height
    dataset = generate_dataset(initial_width, initial_height)

    featurizer = VAEFeaturizer(initial_width, initial_height, learning_rate=args.learning_rate)

    feature_save_path = args.featurizer_save_path
    if restore_model:
        featurizer.load(feature_save_path), dataset
    else:
        featurizer.train(dataset, args.num_epochs, args.batch_size)
        if feature_save_path:
            featurizer.save(feature_save_path)

    return featurizer, dataset


if __name__ == '__main__':
    train()
