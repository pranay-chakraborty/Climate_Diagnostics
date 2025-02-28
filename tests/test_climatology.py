
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from climate_diagnostics_package.climatology import ClimatologyPlotter
import netCDF4

@pytest.fixture
def mock_dataset(tmp_path):
    """
    Creates a mock NetCDF file fortesting purposes.
    """
    time = pd.date_range("1990-01-01",periods=10,freq="ME")
    lat = np.linspace(-90,90,10)
    lon = np.linspace(0,360,10)
    level = np.linspace(1000,100,10)
    
    dataset = xr.Dataset(
        
        {
            "temperature" : (['time','lat','lon','level'],np.random.rand(10,10,10,10))
        },
        coords = {
            "time" : time,
            "lat" : lat,
            "lon" : lon,
            "level" : level,
        },
        
    )
    
    file_path = tmp_path / "mock_dataset.nc"
    dataset.to_netcdf(file_path)
    return str(file_path)

    
def test_init(mock_dataset):
    """
    Checks if the data is initialized correctly via isinstance()
    """
    plots = ClimatologyPlotter(mock_dataset)
    
    assert isinstance(plots.dataset, xr.Dataset)
    assert "temperature" in plots.dataset.data_vars
    
def test_setting_data(mock_dataset):
    """
    Test setting lat, lon, level, time to other values
    """
    plots = ClimatologyPlotter(mock_dataset)
    
    subset = plots.subset_data(lat=slice(-10,10), lon=slice(50,70))
    assert subset.sizes["lat"] == 2
    assert subset.sizes["lon"] == 0
    
    subset_time = plots.subset_data(time = slice('1990-02-28','1990-08-31'))
    assert len(subset_time.time) == 7
    
def test_mean(mock_dataset):
    """
    Tests computing the mean over specified dimensions    
    """
    plots = ClimatologyPlotter(mock_dataset)
    
    spatial_mean = plots.compute_mean(plots.dataset,dim = ['lon','lat'])
    assert 'lon' not in spatial_mean.dims
    assert 'lat' not in spatial_mean.dims

def test_anomaly(mock_dataset):
    """ 
    Test computing the anomalies
    """
    plots = ClimatologyPlotter(mock_dataset)
    anomalies = plots.compute_anomalies(plots.dataset, variable="temperature", dim="time")
    assert anomalies.shape == plots.dataset['temperature'].shape
    
    
def test_plot_trend(mock_dataset, mocker):
    """
    Test that the plot_trend function runs without errors.
    """
    plotter = ClimatologyPlotter(mock_dataset)

    mocker.patch("matplotlib.pyplot.show")

    
    plotter.plot_trend(
        plotter.dataset,
        variable="temperature",
        title="Test Temperature Trend",
        color="blue",
        linestyle="-",
        linewidth=1.5,
    )
    
    plotter.plot_trend(
        plotter.dataset,
        variable="temperature",
        title="Test Temperature Trend",
        color="blue",
        linestyle="--",
        linewidth=1.5,
    )

   
    with pytest.raises(ValueError):
        plotter.plot_trend(
            plotter.dataset,
            variable="invalid_var",
            title="Invalid Variable Trend",
        )