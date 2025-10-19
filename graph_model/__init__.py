"""
Graph Model Package

This package contains the core graph representation and H5 storage functionality
for the Physics-Informed Graph Neural Network (PIGNN) project.

Modules:
    - h5_format_spec: H5 file format specification
    - h5_writer: Write graph data to H5 format
    - graph_exporter: Export graph models to H5
"""

__version__ = "1.0.0"
__author__ = "PIGNN Project"

from .h5_writer import PowerGridH5Writer
from .graph_exporter import GraphToH5Exporter

__all__ = [
    'PowerGridH5Writer',
    'GraphToH5Exporter',
]
