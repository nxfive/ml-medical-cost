from src.optuna.types import LoadedModelResults, ModelRun


class BestRunSelector:
    def __init__(self, runs: LoadedModelResults):
        self.runs = runs

    def select(self) -> ModelRun:
        """
        Selects run with the highest test R2.
        """
        best_score = float("-inf")
        best_run = None

        for model_run in self.runs.runs.values():
            r2 = model_run.result.metrics.get("test_r2", float("-inf"))
            if r2 > best_score:
                best_score = r2
                best_run = model_run

        return best_run
