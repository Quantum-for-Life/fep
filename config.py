import tomllib
import logging
from typing import List
from pydantic import BaseModel, ValidationError, field_validator
from pydantic import NonNegativeFloat, NonNegativeInt, FilePath
from openmm.unit import kelvin, atmospheres, femtoseconds


class SystemSettings(BaseModel):
    file_name: FilePath
    smiles: str
    temperature: NonNegativeFloat
    pressure: NonNegativeFloat
    time_step: NonNegativeFloat
    equilibration_per_window: NonNegativeInt
    sampling_per_window: NonNegativeInt
    sterics_lambdas: List[NonNegativeFloat]
    electrostatics_lambdas: List[NonNegativeFloat]

    @field_validator("temperature")
    @classmethod
    def convert_to_kelvin(cls, v: NonNegativeFloat) -> NonNegativeFloat:
        return v * kelvin

    @field_validator("pressure")
    @classmethod
    def convert_to_atm(cls, v: NonNegativeFloat) -> NonNegativeFloat:
        return v * atmospheres

    @field_validator("time_step")
    @classmethod
    def convert_to_femtoseconds(cls, v: NonNegativeFloat) -> NonNegativeFloat:
        return v * femtoseconds


def load_config(file_path) -> SystemSettings:
    """Load and validate configuration from a TOML file using Pydantic."""
    try:
        with open(file_path, "rb") as file:
            config_data = tomllib.load(file)
            if "system-settings" in config_data:
                return SystemSettings(**config_data["system-settings"])
            else:
                raise ValueError("missing system-settings in config file")
    except ValidationError as e:
        logging.error("Configuration validation error: %s", e.errors())
        raise
