from pathlib import Path

from src.data.converters import CSVToParquetConverter
from src.data.core import DataSaver
from src.data.data import Data
from src.data.download import DatasetDownloader, KaggleDownloader
from src.io.readers import CSVReader, ParquetReader
from src.io.writers import ParquetWriter


class DataFactory:
    @staticmethod
    def create(
        raw_dir: Path, processed_dir: Path, kaggle_handle: str, kaggle_filename: str
    ):
        """
        Creates and wires all objects needed for the data pipeline.
        """
        parquet_reader = ParquetReader()
        csv_reader = CSVReader()
        parquet_writer = ParquetWriter()
        kaggle_downloader = KaggleDownloader(kaggle_handle, kaggle_filename)
        downloader = DatasetDownloader(raw_dir, kaggle_downloader)
        converter = CSVToParquetConverter(csv_reader, parquet_writer)
        data_saver = DataSaver(processed_dir, parquet_writer)

        return Data(raw_dir, parquet_reader, downloader, converter, data_saver)
