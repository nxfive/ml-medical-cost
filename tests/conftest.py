import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    return pd.DataFrame({"age": [20, 30], "bmi": [18.5, 22.8]})


@pytest.fixture
def train_data():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y_train = pd.Series([10, 20, 30, 40, 50])
    return X_train, y_train


class FakeTrial:
    def __init__(self, prune=False):
        self.reported = []
        self._prune = prune
    def suggest_categorical(self, name, choices): return choices[0]
    def suggest_int(self, name, min, max, step=1): return min
    def suggest_float(self, name, min, max, step=1.0): return min
    def report(self, value, step): self.reported.append((value, step))
    def should_prune(self): return self._prune


@pytest.fixture
def fake_trial():
    return FakeTrial()

@pytest.fixture
def fake_trial_prune():
    return FakeTrial(prune=True)


class FakeCV:
    def split(self, X, y):
        return [(slice(0,2), slice(2,4))]

@pytest.fixture
def fake_cv():
    return FakeCV()  