import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from climate_diagnostics_package.climatology import ClimatologyPlotter
else:
    from .climatology import ClimatologyPlotter

__version__ = "0.1.0"
__all__ = ["ClimatologyPlotter"]

