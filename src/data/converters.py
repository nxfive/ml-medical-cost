from pathlib import Path

from src.io.readers import CSVReader
from src.io.writers import ParquetWriter


class CSVToParquetConverter:
    def __init__(self, csv_reader: CSVReader, parquet_writer: ParquetWriter):
        self.csv_reader = csv_reader
        self.parquet_writer = parquet_writer

    def convert(self, csv_path: Path, parquet_path: Path) -> Path:
        """
        Converts a CSV file at the given path into a Parquet file.
        """
        df = self.csv_reader.read(csv_path)
        self.parquet_writer.write(df, parquet_path)
        return parquet_path
