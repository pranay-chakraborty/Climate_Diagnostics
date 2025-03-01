import pytest
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from climate_diagnostics_package.climatology import ClimatologyPlotter

@pytest.fixture
def sample_dataset():
    """Create a sample xarray dataset for testing"""
    time = pd.date_range('2020-01-01', periods=10)
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    level = np.array([1000, 850, 700, 500, 300])
    
    temp_data = np.random.rand(len(time), len(level), len(lat), len(lon)) * 30 + 270  # temperature in K
    precip_data = np.random.rand(len(time), len(lat), len(lon)) * 10  # precipitation in mm/day
    
    ds = xr.Dataset(
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
    
    return ds

@pytest.fixture
def mock_plotter(sample_dataset):
    """Create a mock plotter using a sample dataset"""
    with patch('xarray.open_dataset', return_value=sample_dataset):
        plotter = ClimatologyPlotter("dummy_file_path.nc")
        yield plotter

class TestClimatologyPlotter:
    
    def test_initialization(self, mock_plotter, sample_dataset):
        """Test that the plotter initializes correctly with the dataset"""
        assert set(mock_plotter.dataset.dims) == set(sample_dataset.dims)
        for var in sample_dataset.data_vars:
            assert var in mock_plotter.dataset.data_vars

    def test_subset_data(self, mock_plotter):
        """Test subsetting data by different dimensions"""
        # Test subsetting by latitude
        lat_subset = mock_plotter.subset_data(lat=slice(0, 90))
        assert all(lat_subset.lat >= 0)
        
        # Test subsetting by longitude
        lon_subset = mock_plotter.subset_data(lon=slice(-90, 90))
        assert all(lon_subset.lon >= -90) and all(lon_subset.lon <= 90)
        
        # Test subsetting by level
        level_subset = mock_plotter.subset_data(level=1000)
        assert level_subset.level.item() == 1000
        
        # Test subsetting by time
        time_subset = mock_plotter.subset_data(time=mock_plotter.dataset.time[0])
        assert time_subset.time.size == 1

    def test_compute_mean(self, mock_plotter, sample_dataset):
        """Test computing means over different dimensions"""
        # Test mean over time
        time_mean = mock_plotter.compute_mean(sample_dataset, dim='time')
        assert 'time' not in time_mean.dims
        
        # Test mean over multiple dimensions
        multi_dim_mean = mock_plotter.compute_mean(sample_dataset, dim=['time', 'lat'])
        assert 'time' not in multi_dim_mean.dims
        assert 'lat' not in multi_dim_mean.dims

    def test_compute_anomalies(self, mock_plotter, sample_dataset):
        """Test computing anomalies"""
        anomalies = mock_plotter.compute_anomalies(sample_dataset, 'temperature')
        assert np.abs(anomalies.sum(dim='time').mean().item()) < 1e-10

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    def test_plot_trend(self, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_show, mock_plotter):
        """Test plotting trend runs without error"""
        mock_plotter.plot_trend('temperature', 'time')
        mock_show.assert_called_once()
        mock_title.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    def test_plot_variable(self, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_show, mock_plotter):
        """Test plotting variable runs without error"""
        mock_plotter.plot('temperature')
        mock_show.assert_called_once()
        mock_title.assert_called_once()

    def test_plot_trend_invalid_variable(self, mock_plotter):
        """Test that plotting trend with invalid variable raises error"""
        with pytest.raises(ValueError, match="Variable 'nonexistent_variable' not found"):
            mock_plotter.plot_trend('nonexistent_variable', 'time')

    def test_plot_trend_invalid_dim(self, mock_plotter):
        """Test that plotting trend with invalid dimension raises error"""
        with pytest.raises(ValueError, match="Dimension 'nonexistent_dim' not found"):
            mock_plotter.plot_trend('temperature', 'nonexistent_dim')

    def test_plot_variable_invalid(self, mock_plotter):
        """Test that plotting invalid variable raises error"""
        with pytest.raises(ValueError, match="Variable 'nonexistent_variable' not found"):
            mock_plotter.plot('nonexistent_variable')