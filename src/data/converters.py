from pathlib import Path


class CSVToParquetConverter:
    def __init__(self, csv_reader, parquet_writer):
        self.csv_reader = csv_reader
        self.parquet_writer = parquet_writer

    def convert(self, csv_path: Path, parquet_path: Path):
        df = self.csv_reader.read(csv_path)
        self.parquet_writer.write(df, parquet_path)
        return parquet_path
