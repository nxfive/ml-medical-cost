from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BasePipeline(ABC, Generic[T]):
    @abstractmethod
    def build(self) -> None: ...

    @abstractmethod
    def run(self) -> T: ...
