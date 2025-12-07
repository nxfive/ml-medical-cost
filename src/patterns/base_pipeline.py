from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
B = TypeVar("B")


class BasePipeline(ABC, Generic[B, T]):
    @abstractmethod
    def build(self) -> B: ...

    @abstractmethod
    def run(self) -> T: ...
