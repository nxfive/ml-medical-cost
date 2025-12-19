from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import TransformerMixin

from .identity import IdentityTransformer


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    spec_class: Callable[..., TransformerMixin]
    spec_params: dict[str, Any]

    @property
    def is_identity(self) -> bool:
        return self.spec_class is IdentityTransformer
