from src.containers.results import LoadedModelResults, StageResult


class BestRunSelector:
    def __init__(self, results: LoadedModelResults):
        self.runs = results.runs

    def select(self) -> StageResult:
        """
        Selects run with the highest test R2.
        """
        best_score = float("-inf")
        best_run = None

        for model_run in self.runs.values():
            r2 = model_run.metrics.get("test_r2", float("-inf"))
            if r2 > best_score:
                best_score = r2
                best_run = model_run

        return best_run
