from dataclasses import dataclass

from src.io.readers import CSVReader, JoblibReader, ParquetReader, YamlReader
from src.io.writers import JoblibWriter, ParquetWriter, YamlWriter


@dataclass
class Readers:
    csv: CSVReader
    parquet: ParquetReader
    yaml: YamlReader
    joblib: JoblibReader


@dataclass
class Writers:
    parquet: ParquetWriter
    yaml: YamlWriter
    joblib: JoblibWriter
