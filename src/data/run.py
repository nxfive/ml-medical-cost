from pathlib import Path

from omegaconf import DictConfig

from .io import DataReaderWriter, DatasetDownloader
from .pipeline import DataPipeline


def run_pipeline(cfg: DictConfig):
    downloader = DatasetDownloader()
    io_rw = DataReaderWriter()
    pipeline = DataPipeline(
        downloader,
        io_rw,
        raw_dir=Path(cfg.data.raw_dir),
        processed_dir=Path(cfg.data.processed_dir),
    )
    pipeline()
