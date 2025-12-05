from src.io.readers import CSVReader, JoblibReader, ParquetReader, YamlReader
from src.io.types import Readers, Writers
from src.io.writers import JoblibWriter, ParquetWriter, YamlWriter


class IOFactory:
    @staticmethod
    def create_writers() -> Writers:
        return Writers(
            parquet=ParquetWriter(), yaml=YamlWriter(), joblib=JoblibWriter()
        )

    @staticmethod
    def create_readers() -> Readers:
        return Readers(
            csv=CSVReader(),
            parquet=ParquetReader(),
            yaml=YamlReader(),
            joblib=JoblibReader(),
        )
