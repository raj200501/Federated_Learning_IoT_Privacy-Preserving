"""Compatibility wrapper for synthetic data generation.

This module mirrors data_generation.synthetic_data.generate_synthetic_data so that
existing imports (from synthetic_data...) keep working when running scripts from
repo root.
"""

from data_generation.synthetic_data.generate_synthetic_data import (  # noqa: F401
    generate_iot_data,
    generate_iot_data_configurable,
)

__all__ = ["generate_iot_data", "generate_iot_data_configurable"]
