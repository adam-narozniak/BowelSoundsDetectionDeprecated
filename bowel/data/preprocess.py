import os
import argparse
import csv

import yaml
import pandas as pd
import numpy as np

from bowel.utils.audio_utils import get_wav_length, load_samples, save_wav


class DataProcessor:
    """A class to preprocess raw data.
    """
    def __init__(self, raw_dir, interim_dir, processed_dir, config):
        """DataProcessor constructor.

        Args:
            raw_dir (str): Path to directory with raw data.
            interim_dir (str): Path to directory to save interim data.
            processed_dir (str): Path to directory to save processed data.
            config (dict): Dictionary with config parameters.
        """
        self.config = config
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir

    def process_data(self):
        """Process data from raw to processed.
        """
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.csv'):
                input_dict = os.path.join(self.raw_dir, filename)
                input_wav = input_dict.replace('.csv', '.wav')
                if not os.path.isfile(input_wav):
                    continue
                self.process_file(input_dict, input_wav)
        self.generate_csv_file()

    def process_file(self, input_dict, input_wav):
        """Process single audio file from raw directory

        Args:
            input_dict (str): Path to annotations of audio file.
            input_wav (str): Path to audio file.
        """
        input = pd.read_csv(input_dict)
        length = get_wav_length(input_wav)
        for i in range(int(length // self.config['wav_sample_length'])):
            start = i * self.config['wav_sample_length']
            end = start + self.config['wav_sample_length']
            wav_filename = f'{i}_' + os.path.basename(input_wav)
            output_wav = os.path.join(self.processed_dir, wav_filename)
            output_csv = output_wav.replace('.wav', '.csv')
            samples, sample_rate = load_samples(
                input_wav, offset=start, duration=end - start)
            save_wav(output_wav, samples, sample_rate)
            sounds = input[(input['end'] > start) &
                           (input['start'] < end)].copy()
            sounds[['start', 'end']] -= start
            sounds.to_csv(output_csv, index=False, na_rep='NaN')

    def generate_csv_file(self):
        """Generate csv file with processed files.
        """
        processed_files = os.listdir(self.processed_dir)
        processed_dicts = [f for f in processed_files if f.endswith('.csv')]
        split = np.array_split(processed_dicts, self.config['kfold'])
        output_csv = os.path.join(self.processed_dir, 'files.csv')
        with open(output_csv, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['filename', 'kfold', 'sounds_amount'])
            for i, files in enumerate(split):
                for f_dict in files:
                    f_wav = f_dict.replace('.csv', '.wav')
                    sounds_amount = len(pd.read_csv(os.path.join(self.processed_dir, f_dict)).index)
                    writer.writerow([f_wav, i + 1, sounds_amount])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Path to raw directory')
    parser.add_argument('--interim_dir', type=str, default='data/interim',
                        help='Path to interim directory')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Path to processed directory')
    parser.add_argument('--config', type=str, default='bowel/config.yml',
                        help='Path to yaml file with data and model parameters')
    args = parser.parse_args()
    np.random.seed(10)
    data_processor = DataProcessor(
        args.raw_dir, args.interim_dir, args.processed_dir, yaml.safe_load(open(args.config)))
    data_processor.process_data()
