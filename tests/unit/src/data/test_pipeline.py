from unittest import mock

from src.data.pipeline import run


def test_run(cfg_data):
    with (
        mock.patch("src.data.pipeline.fetch_data") as mock_fetch,
        mock.patch("src.data.pipeline.convert_features_type") as mock_convert,
        mock.patch("src.data.pipeline.split_data") as mock_split,
    ):
        run(cfg_data)

        mock_fetch.assert_called_once_with(cfg_data)
        mock_convert.assert_called_once_with(mock_fetch.return_value)
        mock_split.assert_called_once_with(mock_convert.return_value, cfg_data)
