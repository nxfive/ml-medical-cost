from abc import ABC, abstractmethod
from typing import Generic

from src.containers.types import BuildResultType, SaverType, StageConfigType
from src.data.core import DataLoader, DataSaver
from src.factories.io_factory import IOFactory
from src.factories.model_factory import ModelFactory
from src.models.spec import ModelSpec


class BasePipelineBuilder(ABC, Generic[StageConfigType, SaverType, BuildResultType]):
    def __init__(self, cfg: StageConfigType):
        self.cfg = cfg

    @staticmethod
    def _build_data_io() -> tuple[DataLoader, DataSaver]:
        """
        Builds the DataLoader and DataSaver using IOFactory readers/writers.
        """
        readers = IOFactory.create_readers()
        writers = IOFactory.create_writers()
        return DataLoader(readers), DataSaver(writers)

    def _build_model_spec(self) -> ModelSpec:
        """
        Builds the model specification from the ModelFactory.
        """
        return ModelFactory.get_spec(self.cfg.model.name)

    @abstractmethod
    def _build_saver(self, data_saver: DataSaver) -> SaverType: ...

    @abstractmethod
    def build(self) -> BuildResultType: ...
