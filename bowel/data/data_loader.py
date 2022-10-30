import pathlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from loguru import logger

from bowel.utils.audio_utils import load_samples


class DataLoader:
    """
    Loads divided audio-files of a common length.
    """

    def __init__(self,
                 data_dir: pathlib.Path,
                 annotations_file_path: pathlib.Path) -> None:
        """
        Args:
            data_dir: path to the directory of the preprocessed (divided into common length chunks) audio
            annotations_file_path: file with audio names and number of the fold
        """
        self.data_dir = data_dir
        self.annotations_file_path = annotations_file_path
        # rearrange the columns' data order (for desired MultiIndex of that order)
        self.annotations = pd.read_csv(self.annotations_file_path).iloc[:, :-1][["kfold", "filename"]]

    def load_data(self, multiprocessing: int = 4) -> pd.DataFrame:
        """
        Loads the data specified in annotations file.
        Args:
            multiprocessing: number of running threads

        Returns:
            pd.DataFrame with "kfold" and "filename" index pointing to single channel audio-file.
        """
        logger.info("Data loading started")
        pool = Pool(multiprocessing)
        data = pool.map(self._load_single_file, self.annotations.index.values)
        index, audios = zip(*data)
        index = pd.MultiIndex.from_frame(pd.DataFrame(index), names=["kfold", "filename"])
        logger.info("Data loading done")
        return pd.DataFrame(audios, index=index, columns=["audio"])

    def _load_single_file(self, index: int) -> tuple[list[int, str], list[np.array]]:
        """
        Based on the absolute index (0 to the number of files) of the annotation file load an audio sample.
        Args:
            index: absolute index from the file describing the how the audio was divided

        Returns:
            id and filename of the loaded audio, audio (pcm values)
        """
        audio_id_and_filename = self.annotations.loc[index]
        filename = audio_id_and_filename["filename"]
        audio_path = self.data_dir / filename
        audio = load_samples(audio_path)[0]
        return audio_id_and_filename.tolist(), [audio]


