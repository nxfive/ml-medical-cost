from pathlib import Path

from src.data.converters import CSVToParquetConverter
from src.data.core import DataFetcher, DataSaver
from src.data.data import Data
from src.data.download import DatasetDownloader, KaggleDownloader

from .io_factory import IOFactory


class DataFactory:
    @staticmethod
    def create(
        raw_dir: Path, processed_dir: Path, kaggle_handle: str, kaggle_filename: str
    ) -> Data:
        """
        Creates and wires all objects needed for the data pipeline.
        """
        readers = IOFactory.create_readers()
        writers = IOFactory.create_writers()

        kaggle_downloader = KaggleDownloader(kaggle_handle, kaggle_filename)
        downloader = DatasetDownloader(raw_dir, kaggle_downloader)
        converter = CSVToParquetConverter(readers.csv, writers.parquet)
        data_saver = DataSaver(writers)
        data_fetcher = DataFetcher(raw_dir, downloader, converter, readers)

        return Data(data_saver, data_fetcher, processed_dir)
