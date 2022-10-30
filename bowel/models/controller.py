import argparse
import os
import pathlib

import flwr as fl
import keras
import numpy as np
import wandb
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Precision, Recall
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from wandb.keras import WandbCallback

from bowel.config import WANDB_PROJECT_NAME
from bowel.data.data_loader import DataLoader
from bowel.data.label_creator import LabelCreator
from bowel.data.normalizers.all_normalizer import AllNormalizer
from bowel.data.normalizers.mfcc_rms_normalizer import MfccRmsNormalizer
from bowel.data.transformers.audio_features_transformer import AudioFeaturesTransformer
from bowel.data.transformers.mean_std_transformer import MeanStdTransformer
from bowel.data.transformers.raw_audio_transformer import RawAudioTransformer
from bowel.data.transformers.spectrogram_transformer import SpectrogramTransformer
from bowel.fl.client import Client
from bowel.models.lstm_model import LSTM_model
from bowel.models.lstm_with_conv_model import LSTM_with_conv_model
from bowel.utils.io_utils import load_config, save_yaml
from bowel.utils.reproducibility_utils import setup_seed
from bowel.utils.train_utils import get_score, get_scores_mean


class Controller:
    """Trains, test or cross-validates models."""

    def __init__(self,
                 data_dir: pathlib.Path,
                 division_file_path: pathlib.Path,
                 transform_config_path: pathlib.Path,
                 train_config_path: pathlib.Path):
        self._data_dir = data_dir
        self._division_file_path = division_file_path
        self._transform_config_path = transform_config_path
        self._transform_config = load_config(transform_config_path)
        self._train_config = load_config(train_config_path)
        self.full_config = {**self._train_config, **self._transform_config}
        self._kfold_list = list(range(1, self._train_config["kfold"] + 1))
        self._model_type = self._train_config["model_type"]
        self.model = None
        self._features_type = self._transform_config["features_type"]
        self._data_loader = DataLoader(data_dir, division_file_path)
        self._data = self._data_loader.load_data()
        self._label_creator = LabelCreator(data_dir, division_file_path, self._transform_config)
        self._labels = self._label_creator.labels
        self.data_transformer = None
        self._audio_normalizer = None
        self._features_normalizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_history = None
        self.train_evaluation_score = None
        self.test_evaluation_score = None

    def instantiate_model(self):
        """Creates model based on the parameters specified in configuration file."""
        if self._model_type == "mfcc_lstm":
            self.model = LSTM_model()
        elif self._model_type == "spec_lstm":
            self.model = LSTM_model()
        elif self._model_type == "spec_lstm_with_conv":
            self.model = LSTM_with_conv_model()
        elif self._model_type == "dummy":
            self.model = DummyClassifier(strategy="most_frequent")
        elif self._model_type == "mean_std_reg":
            self.model = LogisticRegression(solver="saga",
                                            max_iter=self._train_config["epochs"],
                                            class_weight=self._train_config["class_weight"])
        else:
            raise KeyError()

        if isinstance(self.model, keras.Model):
            self.model.compile(loss=self._train_config["loss"], optimizer=self._train_config["optimizer"],
                               metrics=['accuracy', Precision(), Recall(), AUC(curve='PR')])

    def instantiate_normalizers(self, mode: str = None, save_path=None):
        """Creates normalizers to be adapted or uses previously computed mean and variance (test mode)"""
        if mode == "test":
            config = load_config(save_path / "norm.yaml")
            self._audio_normalizer = AllNormalizer(normalize=self.full_config["normalize_audio"],
                                                   mean=config["audio_mean"], variance=config["audio_var"])
            if self._features_type == "mfcc":
                self._features_normalizer = MfccRmsNormalizer(normalize=self.full_config["normalize_features"],
                                                              mean_mfcc=config["mean_mfcc"],
                                                              variance_mfcc=config["var_mfcc"],
                                                              mean_rms=config["mean_rms"],
                                                              variance_rms=config["var_rms"])
            else:
                self._features_normalizer = AllNormalizer(normalize=self.full_config["normalize_features"],
                                                          mean=config["feature_mean"],
                                                          variance=config["feature_variance"])
        else:
            self._audio_normalizer = AllNormalizer(normalize=self.full_config["normalize_audio"])
            self._features_normalizer = MfccRmsNormalizer(normalize=self.full_config["normalize_features"])

    def _instantiate_transformer(self, data):
        """Creates object to change audio files into features for a neural network."""
        if self._features_type == "mfcc":
            self.data_transformer = AudioFeaturesTransformer(data, self._transform_config)
        elif self._features_type == "spec" and self._model_type != "spec_lstm_with_conv":
            self.data_transformer = SpectrogramTransformer(data, self._transform_config)
        elif self._features_type == "spec":
            self.data_transformer = SpectrogramTransformer(data, self._transform_config, expand_dims=True)
        elif self._features_type == "raw":
            self.data_transformer = RawAudioTransformer(data, self._transform_config)
        elif self._features_type == "mean_std":
            self.data_transformer = MeanStdTransformer(data, self._transform_config)
        else:
            raise KeyError()

    def _create_train_test_mask(self, test_fold):
        """Based on the folds (given in the data) creates masks ready to used for indexing."""
        kfold_list = self._kfold_list.copy()
        kfold_list.remove(test_fold)
        train_kfold_list = kfold_list
        train_mask = np.isin(self._data.index.get_level_values('kfold').values, train_kfold_list)
        test_mask = self._data.index.get_level_values('kfold').values == test_fold
        return train_mask, test_mask

    def create_train_test_data(self, test_fold=None):
        """Given loaded data (optionally) normalizes audio and transforms into the features and (optionally)
        normalizes features """
        if test_fold is None:
            test_fold = self._kfold_list[-1]
        train_mask, test_mask = self._create_train_test_mask(test_fold)
        train_part = int(train_mask.shape[0] * (1. - self._train_config["validation_split"]))
        data = np.array([row[0] for row in self._data.values])
        self._audio_normalizer.adapt(data[train_mask][:train_part])
        data[train_mask] = self._audio_normalizer.normalize(data[train_mask])
        data[test_mask] = self._audio_normalizer.normalize(data[test_mask])
        self._instantiate_transformer(data)
        transformed_data = self.data_transformer.transformed
        self._features_normalizer.adapt(transformed_data)
        transformed_data[train_mask] = self._features_normalizer.normalize(transformed_data[train_mask])
        transformed_data[test_mask] = self._features_normalizer.normalize((transformed_data[test_mask]))
        if self._transform_config["subtimesteps"]:
            pass
        else:
            train_mask = np.repeat(train_mask, self._label_creator.n_substamps)
            test_mask = np.repeat(test_mask.reshape(-1, 1), self._label_creator.n_substamps)
        self.X_train, self.X_test = transformed_data[train_mask], transformed_data[test_mask]
        self.y_train, self.y_test = self._labels[train_mask], self._labels[test_mask]

    def train_test(self):
        """Firstly trains the data on the first n-1 folds, then tests it on the remaining one fold."""
        self.instantiate_model()
        self.train_evaluation_score = self.train()
        self.test_evaluation_score = self.test()

    def crossval(self):
        train_scores = []
        test_scores = []
        for test_fold in self._kfold_list[:-1]:  # the last "fold" is the test set
            self.instantiate_model()
            self.create_train_test_data(test_fold)
            train_score = self.train()
            train_scores.append(train_score)
            test_score = self.test()
            test_scores.append(test_score)
        print("Averaged train scores")
        self.train_evaluation_score = get_scores_mean(train_scores)
        print(self.train_evaluation_score)
        print("Averaged test scores")
        self.test_evaluation_score = get_scores_mean(test_scores)
        print(self.test_evaluation_score)

    def train(self):
        if self._train_config["library"] == "keras":
            self.train_history = self.model.fit(self.X_train,
                                                self.y_train,
                                                validation_split=self._train_config["validation_split"],
                                                epochs=self._train_config["epochs"],
                                                callbacks=[
                                                    WandbCallback(save_model=False),
                                                    EarlyStopping(
                                                        monitor="val_loss",
                                                        patience=self._train_config["patience"],
                                                        restore_best_weights=True
                                                    )])
        elif self._train_config["library"] == "sklearn":
            self.model.fit(self.X_train,
                           self.y_train)
        else:
            raise KeyError()
        y_train_pred = self.model.predict(self.X_train)
        print("X train:")
        train_evaluation_score = get_score(self.y_train, y_train_pred)
        print(train_evaluation_score)
        return train_evaluation_score

    def test(self):
        y_test_pred = self.model.predict(self.X_test)
        print("X test:")
        test_evaluation_score = get_score(self.y_test, y_test_pred)
        print(test_evaluation_score)
        return test_evaluation_score

    def test_from_saved(self, test_fold=None, save_path: pathlib.Path = None):
        if test_fold is None:
            test_fold = self._kfold_list[-1]
        train_mask, test_mask = self._create_train_test_mask(test_fold)
        data = np.array([row[0] for row in self._data.values])
        X_test = self._audio_normalizer.normalize(data[test_mask]).numpy()
        self._instantiate_transformer(X_test)
        transformed_data = self.data_transformer.transformed
        transformed_data = self._features_normalizer.normalize(transformed_data)
        self.X_test = transformed_data
        self.y_test = self._labels[test_mask]
        self.model = keras.models.load_model(save_path)
        self.test()

    def train_only(self):
        self.instantiate_model()
        self.train()

    def save_model(self, path: pathlib.Path):
        if self._train_config["library"] == "keras":
            keras.models.save_model(self.model, path)
            self._save_norm_params(path / "norm.yaml")
        elif self._train_config["library"] == "sklearn":
            pass
        else:
            raise KeyError()

    def _save_norm_params(self, path: pathlib.Path):
        """Mean and variance used in the normalization need to be saved for saved model use (models from a disc)."""
        normalizer_params = {
            "audio_mean": float(self._audio_normalizer.normalizer.mean.numpy()[0]),
            "audio_var": float(self._audio_normalizer.normalizer.variance.numpy()[0])
        }
        if isinstance(self._features_normalizer, MfccRmsNormalizer):
            normalizer_params["mean_mfcc"] = float(self._features_normalizer.mfcc_normalizer.mean.numpy()[0])
            normalizer_params["var_mfcc"] = float(self._features_normalizer.mfcc_normalizer.variance.numpy()[0])
            normalizer_params["mean_mfcc"] = float(self._features_normalizer.rms_normalizer.mean.numpy()[0])
            normalizer_params["var_mfcc"] = float(self._features_normalizer.rms_normalizer.variance.numpy()[0])
        save_yaml(normalizer_params, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,
                        help="{'train'| 'train_test' | 'crossval' | 'test' | 'federated_client'}")
    parser.add_argument("--data_dir", type=pathlib.Path, default=pathlib.Path("./data/processed/"),
                        help="path to data directory")
    parser.add_argument("--division_file", type=pathlib.Path, default=pathlib.Path("./data/processed/files.csv"),
                        help="path to the file specifying samples names and division in folds")
    parser.add_argument("--trans_config", type=str,
                        help="name of the file to the transformation configuration; "
                             "file must be place in ./configs dir")
    parser.add_argument("--train_config", type=str,
                        help="name of the file to the training configuration;"
                             "file must be place in ./configs dir")
    parser.add_argument("-s", "--save_path", type=pathlib.Path, default=None,
                        help="path to a dir to save (or load, depending on the mode) the model; "
                             "model is not saved if left None")
    parser.add_argument("-l", "--log", action="store_true",
                        help="add this flag if you want to use wandb."
                             "if left empty, the experiment won't be logged online, instead offline mode will be used")
    parser.add_argument("-n", "--wandb_log_name", type=str, default="",
                        help="name to log the experiment on wandb (ignored if -l flag is not set);"
                             "if not give, a random name will appear")
    setup_seed()
    args = parser.parse_args()

    config_dir = pathlib.Path("./configs/")
    if_log = args.log
    log_name = args.wandb_log_name
    controller = Controller(
        args.data_dir,
        args.division_file,
        config_dir / args.trans_config,
        config_dir / args.train_config
    )
    mode = args.mode
    save_path = args.save_path
    if not if_log:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=WANDB_PROJECT_NAME, config=controller.full_config)
    if log_name != "":
        wandb.run.name = log_name
    if mode == "train":
        controller.instantiate_normalizers()
        controller.create_train_test_data()
        controller.train_only()
        for k, v in controller.train_evaluation_score.items():
            wandb.run.summary["train_evaluation/" + str(k)] = v
    elif mode == "train_test":
        controller.instantiate_normalizers()
        controller.create_train_test_data()
        controller.train_test()
        for k, v in controller.train_evaluation_score.items():
            wandb.run.summary["train_evaluation/" + str(k)] = v
        for k, v in controller.test_evaluation_score.items():
            wandb.run.summary["test_evaluation/" + str(k)] = v
    elif mode == "crossval":
        controller.instantiate_normalizers()
        controller.crossval()
        for k, v in controller.train_evaluation_score.items():
            wandb.run.summary["avg_train_evaluation/" + str(k)] = v
        for k, v in controller.test_evaluation_score.items():
            wandb.run.summary["avg_test_evaluation/" + str(k)] = v
    elif mode == "test":
        controller.instantiate_normalizers(mode="test", save_path=save_path)
        controller.test_from_saved(save_path=save_path)
    elif mode == "federated_client":
        controller.instantiate_normalizers()
        controller.create_train_test_data()
        controller.instantiate_model()
        fl.client.start_numpy_client(server_address="[::]:8080",
                                     client=Client(controller.model, controller.X_train, controller.y_train,
                                                   controller.X_test, controller.y_test))
    else:
        raise KeyError()
    if (mode == "train" or mode == "train_test") and save_path is not None:
        controller.save_model(save_path)
