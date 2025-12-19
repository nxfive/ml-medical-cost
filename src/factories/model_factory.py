from src.models.registry import MODELS
from src.models.spec import ModelSpec


class ModelFactory:
    @staticmethod
    def get_spec(name_or_alias: str) -> ModelSpec:
        """
        Retrieves the ModelSpec for a given model name or alias.
        """
        for name, spec in MODELS.items():
            if name_or_alias in (name, spec.alias):
                return spec
        raise ValueError(f"Model '{name_or_alias}' not found")
