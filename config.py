import tomllib
import logging
from typing import Annotated
from typing import List
from pydantic import BaseModel, ValidationError, field_validator, Field
from pydantic import NonNegativeFloat, NonNegativeInt, FilePath
from openmm.unit import kelvin, atmospheres, femtoseconds
from openmm import unit


class SystemSettings(BaseModel):
    file_name: FilePath
    smiles: str
    temperature: dict
    pressure: dict
    time_step: dict
    equilibration_per_window: NonNegativeInt
    sampling_per_window: NonNegativeInt
    sterics_lambdas: List[NonNegativeFloat]
    electrostatics_lambdas: List[NonNegativeFloat]

    @field_validator("sterics_lambdas", "electrostatics_lambdas")
    @classmethod
    def verify_lambdas(cls, v: List[NonNegativeFloat]) -> List[NonNegativeFloat]:
        is_valid = all([(x <= 1.0) & (x >= 0.0) for x in v])
        if not is_valid:
            raise ValueError(f"values not between 0.0 and 1.0")
        return v

    @field_validator("temperature", "pressure", "time_step")
    @classmethod
    def validate_physical_unit(cls, v: dict) -> unit.Quantity:
        return unit.Quantity(NonNegativeFloat(v["target"]), getattr(unit, v["unit"]))


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
