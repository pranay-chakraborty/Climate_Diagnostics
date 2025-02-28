
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


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

    def subset_data(self, lat=None, lon=None, level=None, time=None):
        """
        Subset the dataset based on user-selected dimensions.

        Args:
            lat (slice or float): Latitude range or single value.
            lon (slice or float): Longitude range or single value.
            layer (slice or float): Layer (e.g., pressure level) range or single value.
            time (slice or str): Time range or single value.

        Returns:
            xr.Dataset: Subsets the dataset based on the provided dimensions.
        """
        subset = self.dataset
        if lat is not None:
            subset = subset.sel(lat=lat)
        if lon is not None:
            subset = subset.sel(lon=lon)
        if level is not None:
            subset = subset.sel(level=level)  
        if time is not None:
            subset = subset.sel(time=time)
        return subset

    def compute_mean(self, data, dim=None):
        """
        Compute the mean over a specified dimension.

        Args:
            data (xr.Dataset): The dataset to compute the mean for.
            dim (str or list): Dimension(s) to average over.

        Returns:
            xr.Dataset: Dataset with the mean computed over the specified dimension(s).
        """
        # if dim not in self.dataset.dims:
        #     raise ValueError(f"Dimension '{dim}' not found in the dataset.")
        return data.mean(dim=dim)


    def compute_anomalies(self, data, variable, dim="time"):
        """
        Compute anomalies (deviations from the mean) for a specific variable.

        Args:
            data (xr.Dataset): The dataset containing the variable.
            variable (str): The variable to compute anomalies for.
            dim (str): Dimension to calculate anomalies over (default: "time").

        Returns:
            xr.DataArray: Anomalies for the specified variable.
        """
        mean = data[variable].mean(dim=dim)
        anomalies = data[variable] - mean
        return anomalies

    def plot_trend(
        self,
        data,
        variable,
        title="Climatology Trend",
        color="blue",
        linestyle="-",
        linewidth=2,
        xlabel="Time",
        ylabel=None,
    ):
        """
        Plot the trend of a specific variable with customization options.

        Args:
            data (xr.Dataset): The dataset containing the variable.
            variable (str): The variable to plot.
            title (str): Title of the plot.
            color (str): Color of the plot line.
            linestyle (str): Style of the plot line.
            linewidth (float): Width of the plot line.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        if variable not in data.variables:
            raise ValueError(f"Variable '{variable}' not found in the dataset.")

        data[variable].plot(
            color=color, linestyle=linestyle, linewidth=linewidth, label=variable
        )
        plt.title(title)
        plt.xlabel(xlabel)
        if ylabel is None:
            ylabel = variable
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

