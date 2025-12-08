from src.conf.schema import DataStageConfig
from src.data.converters import CSVToParquetConverter
from src.data.core import DataFetcher, DataSaver
from src.data.data import Data
from src.data.download import DatasetDownloader, KaggleDownloader

from .io_factory import IOFactory


class DataFactory:
    @staticmethod
    def create(cfg: DataStageConfig) -> Data:
        """
        Creates and wires all objects needed for the data pipeline.
        """
        readers = IOFactory.create_readers()
        writers = IOFactory.create_writers()

        kaggle_downloader = KaggleDownloader(cfg.kaggle.handle, cfg.kaggle.filename)
        downloader = DatasetDownloader(cfg.data_dir.raw_dir, kaggle_downloader)
        converter = CSVToParquetConverter(readers.csv, writers.parquet)
        data_saver = DataSaver(writers)
        data_fetcher = DataFetcher(cfg.data_dir.raw_dir, downloader, converter, readers)

        return Data(data_saver, data_fetcher, cfg.data_dir.processed_dir)
