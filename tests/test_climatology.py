import pytest
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from climate_diagnostics_package.climatology import ClimatologyPlotter

@pytest.fixture
def sample_dataset():
    """Create a sample xarray dataset for testing"""
    time = pd.date_range('2020-01-01', periods=10)
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    level = np.array([1000, 850, 700, 500, 300])

    temp_data = np.random.rand(10, 5, 5, 5) * 30 + 270  # 10 time steps
    precip_data = np.random.rand(10, 5, 5) * 10

    return xr.Dataset(
        data_vars={
            'temperature': (('time', 'level', 'lat', 'lon'), temp_data),
            'precipitation': (('time', 'lat', 'lon'), precip_data),
        },
        coords={
            'time': time,
            'level': level,
            'lat': lat,
            'lon': lon,
        }
    )

@pytest.fixture
def mock_plotter(sample_dataset):
    """Create a mock plotter using a sample dataset"""
    with patch('xarray.open_dataset', return_value=sample_dataset):
        return ClimatologyPlotter("dummy_file.nc")

class TestClimatologyPlotter:

    def test_initialization(self, mock_plotter, sample_dataset):
        """Test proper initialization with dataset"""
        assert mock_plotter.dataset.dims == sample_dataset.dims
        assert all(var in mock_plotter.dataset for var in sample_dataset.data_vars)

    def test_select(self, mock_plotter):
        """Test data subsetting"""
        # Latitude subset
        subset = mock_plotter.select(lat=slice(0, 90))
        assert all(subset.lat >= 0) and all(subset.lat <= 90)

        # Single level selection
        subset = mock_plotter.select(level=850)
        assert subset.level.values == 850

    def test_mean_calculation(self, mock_plotter):
        """Test mean computation over dimensions"""
        # Time mean
        time_mean = mock_plotter.mean(dimension='time')
        assert 'time' not in time_mean.dims

        # Multi-dimensional mean
        multi_mean = mock_plotter.mean(dimension=['time', 'lat'])
        assert {'time', 'lat'}.isdisjoint(multi_mean.dims)

    def test_anomalies(self, mock_plotter):
        """Test anomaly calculation"""
        anomalies = mock_plotter.anomalies('temperature', dim='time')
        assert np.allclose(anomalies.mean(dim='time'), 0, atol=1e-10)

    @patch('matplotlib.pyplot.show')
    def test_plot_trend(self, mock_show, mock_plotter):
        """Test trend plotting"""
        mock_plotter.plot_trend('temperature', 'time')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_variable(self, mock_show, mock_plotter):
        """Test variable plotting"""
        mock_plotter.plot('temperature')
        mock_show.assert_called_once()

    def test_invalid_variable(self, mock_plotter):
        """Test error handling for invalid variables"""
        with pytest.raises(ValueError, match="not found in the dataset"):
            mock_plotter.anomalies('invalid_var')

        with pytest.raises(ValueError, match="not found in the dataset"):
            mock_plotter.plot('invalid_var')

    def test_invalid_dimension(self, mock_plotter):
        """Test error handling for invalid dimensions"""
        with pytest.raises(ValueError, match="Dimension .* not found"):
            mock_plotter.mean(dimension='invalid_dim')

        with pytest.raises(ValueError, match="Dimension .* not found"):
            mock_plotter.plot_trend('temperature', 'invalid_dim')
