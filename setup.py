from setuptools import setup, find_packages

setup(
    name="climate_diagnostics",
    version="0.1.0",
    author="Pranay Chakraborty", 
    author_email="pranay.chakraborty.personal@gmail.com",
    description="Climate diagnostics tools for analyzing and visualizing climate data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    #url=""
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Runtime dependencies from environment.yml
        "xarray",
        "dask",
        "netCDF4",
        "bottleneck",
        "matplotlib",
        "numpy",
        "scipy",
        "cartopy",
    ],
    extras_require={
        "dev": [
            # Conda-installed dev tools available via pip
            "pytest",
            "flake8",
            "jupyter",
            "ipykernel",
            "mypy",
            "pre-commit",
            # Pip-only dev tools
            "pytest-cov",
            "black",
            "isort",
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
            "tox",
            "build",
            "twine",
        ],
    },
    python_requires=">=3.11",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
)