import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import copy

class ClimatologyPlotter:
    """
    A class to handle climatology plotting from NetCDF datasets.
    """

    def __init__(self, file_path):
        """
        Initialize the plotter with a NetCDF file.

        Args:
            file_path (str): Path to the NetCDF file.
        """
        self.dataset = xr.open_dataset(file_path)
    
    def _filter_dataset(self, dataset, lat=None, lon=None, level=None, time=None):
        """
        Helper method to filter the dataset based on dimensions.

        Args:
            dataset (xr.Dataset): Dataset to filter.
            lat (slice or float): Latitude range or single value.
            lon (slice or float): Longitude range or single value.
            level (slice or float): Layer (e.g., pressure level) range or single value.
            time (slice or str): Time range or single value.

        Returns:
            xr.Dataset: Filtered dataset.
        """
        filtered = dataset
        if lat is not None:
            filtered = filtered.sel(lat=lat)
        if lon is not None:
            filtered = filtered.sel(lon=lon)
        if level is not None:
            filtered = filtered.sel(level=level)  
        if time is not None:
            filtered = filtered.sel(time=time)
        return filtered
        
    def select(self, lat=None, lon=None, level=None, time=None):
        """
        Select the dataset based on user-selected dimensions.

        Args:
            lat (slice or float): Latitude range or single value.
            lon (slice or float): Longitude range or single value.
            level (slice or float): Layer (e.g., pressure level) range or single value.
            time (slice or str): Time range or single value.

        Returns:
            xr.Dataset: Subsets the dataset based on the provided dimensions.
        """
        return self._filter_dataset(self.dataset, lat, lon, level, time)

    def mean(self, dimension=None, lat=None, lon=None, level=None, time=None):
        """
        Compute the mean over specified dimensions.

        Args:
            dimension (str or list of str, optional): Dimension(s) to average over. 
                Can be a single dimension name or a list of dimension names.
                If None, returns the dataset unchanged.
            lat (slice or float, optional): Latitude range or single value.
            lon (slice or float, optional): Longitude range or single value.
            level (slice or float, optional): Level range or single value.
            time (slice or str, optional): Time range or single value.

        Returns:
            xr.Dataset: Dataset with the mean computed over the specified dimension(s).
        """
        # First filter the dataset
        data = self._filter_dataset(self.dataset, lat, lon, level, time)
        
        if dimension is None:
            return data
        
        # Convert single string dimension to a list
        if isinstance(dimension, str):
            dimension = [dimension]
        elif not isinstance(dimension, list):
            raise TypeError("'dimension' must be a string, a list of strings, or None")
        
        # Validate all dimensions
        for d in dimension:
            if not isinstance(d, str):
                raise TypeError(f"All dimensions must be strings, got {type(d)}")
            if d not in data.dims:
                raise ValueError(f"Dimension '{d}' not found in the dataset.")
        
        return data.mean(dim=dimension)

    def anomalies(self, variable, dim="time", lat=None, lon=None, level=None, time=None):
        """
        Compute anomalies (deviations from the mean) for a specific variable.

        Args:
            variable (str): The variable to compute anomalies for.
            dim (str): Dimension to calculate anomalies over (default: "time").
            lat (slice or float, optional): Latitude range or single value.
            lon (slice or float, optional): Longitude range or single value.
            level (slice or float, optional): Level range or single value.
            time (slice or str, optional): Time range or single value.

        Returns:
            xr.DataArray: Anomalies for the specified variable.
        """
        # First filter the dataset
        data = self._filter_dataset(self.dataset, lat, lon, level, time)
        
        if variable not in data.variables:  
            raise ValueError(f"Variable '{variable}' not found in the dataset.")
        
        mean = data[variable].mean(dim=dim)
        anomalies = data[variable] - mean
        return anomalies

    def plot_trend(
        self,
        variable,
        dim,
        lat=None,
        lon=None,
        level=None,
        time=None,
        title=None,
        color="blue",
        linestyle="-",
        linewidth=2,
        xlabel=None,
        ylabel=None,
    ):
        """
        Plot the trend of a specific variable with customization options.

        Args:
            variable (str): The variable to plot.
            dim (str): Dimension to plot the trend along.
            lat (slice or float, optional): Latitude range or single value.
            lon (slice or float, optional): Longitude range or single value.
            level (slice or float, optional): Level range or single value.
            time (slice or str, optional): Time range or single value.
            title (str, optional): Title of the plot. If None, uses default format.
            color (str): Color of the plot line.
            linestyle (str): Style of the plot line.
            linewidth (float): Width of the plot line.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        # First filter the dataset
        data = self._filter_dataset(self.dataset, lat, lon, level, time)
        
        if variable not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in the dataset.")
        
        if dim not in data.dims:
            raise ValueError(f"Dimension '{dim}' not found in the dataset.")
        
        if title is None:
            title = f'Trend of {variable} over {dim}'
        
        dims_to_average = [d for d in data.dims if d != dim]
        
        if dims_to_average:
            tmp_data = data[variable].mean(dim=dims_to_average)
        else:
            tmp_data = data[variable]
            
        tmp_data.plot(color=color, linestyle=linestyle, linewidth=linewidth)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else dim)
        plt.ylabel(ylabel if ylabel else variable)
        plt.legend()
        plt.show()

    def plot(
        self,
        variable,
        lat=None,
        lon=None,
        level=None,
        time=None,
        title=None,
        color="blue",
        linestyle="-",
        linewidth=2,
        xlabel=None,
        ylabel=None
    ):
        """
        Plot a specific variable with customization options.
        
        Args:
            variable (str): The variable to plot.
            lat (slice or float, optional): Latitude range or single value.
            lon (slice or float, optional): Longitude range or single value.
            level (slice or float, optional): Level range or single value.
            time (slice or str, optional): Time range or single value.
            title (str, optional): Title of the plot. If None, uses the variable name.
            color (str): Color of the plot line.
            linestyle (str): Style of the plot line.
            linewidth (float): Width of the plot line.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis. If None, uses the variable name.
        """
        # First filter the dataset
        data = self._filter_dataset(self.dataset, lat, lon, level, time)
        
        if variable not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in the dataset.")
        
        if title is None:
            title = variable
        
        data[variable].plot(
            color=color, linestyle=linestyle, linewidth=linewidth, label=variable
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()