import sys
from pathlib import PosixPath
import argparse
import logging
import config
from config import SystemSettings

# from pymbar import MBAR, timeseries
import mdtraj
from openff.toolkit import Molecule
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PDBReporter,
    PME,
    Simulation,
    StateDataReporter,
)
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtools import alchemy
import openmm
from openmm import (
    CustomNonbondedForce,
    LangevinIntegrator,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
    XmlSerializer,
)
from openmm.unit import (
    AVOGADRO_CONSTANT_NA,
    BOLTZMANN_CONSTANT_kB,
    amu,
    angstrom,
    atmospheres,
    femtoseconds,
    kelvin,
    kilocalories_per_mole,
    nanometer,
    picoseconds,
)


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LambdaScheme:
    def __init__(
        self,
        steric_lambdas,
        electrostatic_lambdas,
    ):
        self._steric_lambdas = steric_lambdas
        # to avoid calculating the same lambda state twice
        if electrostatic_lambdas[-1] == 0.0:
            self._electrostatic_lambdas = electrostatic_lambdas[:-1]
        else:
            self._electrostatic_lambdas = electrostatic_lambdas

        sterics = np.zeros(len(self._steric_lambdas) + len(self._electrostatic_lambdas))
        electrostatics = sterics.copy()
        sterics[: len(self._electrostatic_lambdas)] = 1.0
        sterics[len(self._electrostatic_lambdas) :] = self._steric_lambdas
        electrostatics[: len(self._electrostatic_lambdas)] = self._electrostatic_lambdas
        electrostatics[len(self._electrostatic_lambdas) :] = 0.0
        self._lambda_scheme = list(zip(sterics, electrostatics))

    @property
    def lambda_scheme(self):
        return self._lambda_scheme


def create_alchemical_system(
    molecule_file: openmm.app.pdbfile.PDBFile,
    forcefield: openmm.app.ForceField,
    SystemSettings: config.SystemSettings,
) -> openmm.System:
    modeller = Modeller(molecule_file.topology, molecule_file.positions)
    modeller.addSolvent(forcefield, padding=1.1 * nanometer)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        switchDistance=0.9 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * amu,
    )

    factory = alchemy.AbsoluteAlchemicalFactory()

    small_molecule_atoms = mdtraj.Topology.from_openmm(modeller.topology).select(
        "resname UNL"
    )
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=small_molecule_atoms)
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)

    return alchemical_system


def load_small_molecule(
    file_name: PosixPath, smiles: str, forcefield: openmm.app.ForceField
) -> None:
    if file_name.suffix == ".pdb":
        loaded_molecule = PDBFile(f"{file_name}")
    else:
        raise ValueError(f"file format not implemented: {file_name}")

    molecule = Molecule.from_smiles(f"{smiles}")
    gaff = GAFFTemplateGenerator(molecules=molecule)
    forcefield.registerTemplateGenerator(gaff.generator)

    return loaded_molecule


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Calculate FEP")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    logging.info(f"Configuration loaded and validated: {cfg}")

    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
    loaded_molecule = load_small_molecule(cfg.file_name, cfg.smiles, forcefield)

    create_alchemical_system(loaded_molecule, forcefield, cfg)


if __name__ == "__main__":
    sys.exit(main())
