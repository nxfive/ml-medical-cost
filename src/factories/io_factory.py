from src.io.readers import (BaseReader, CSVReader, JoblibReader, ParquetReader,
                            YamlReader)
from src.io.writers import BaseWriter, JoblibWriter, ParquetWriter, YamlWriter


class IOFactory:
    @staticmethod
    def create_writers() -> dict[str, BaseWriter]:
        return {
            "parquet": ParquetWriter(),
            "yaml": YamlWriter(),
            "joblib": JoblibWriter(),
        }

    @staticmethod
    def create_readers() -> dict[str, BaseReader]:
        return {
            "csv": CSVReader(),
            "parquet": ParquetReader(),
            "yaml": YamlReader(),
            "joblib": JoblibReader(),
        }
