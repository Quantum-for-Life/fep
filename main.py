from typing import Tuple, Iterator
from collections.abc import Iterable
from config import SystemSettings
from pathlib import PosixPath
import argparse
import config
import copy
import logging
import sys
from typing import Type

# from pymbar import MBAR, timeseries
import numpy as np
from openff.toolkit import Molecule
import mdtraj
import openmmtools
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


def setup_logging() -> None:
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

    # @property
    # def lambda_scheme(self):
    #     return self._lambda_scheme

    # def __repr__(self):
    #     return self._lambda_scheme

    def __len__(self) -> int:
        return len(self._lambda_scheme)

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for each in self._lambda_scheme:
            yield each

    def __getitem__(self, index: int) -> Tuple[float, float]:
        return self._lambda_scheme[index]


class AlchemicalSystem:
    def __init__(
        self,
        molecule_file: openmm.app.PDBFile,
        forcefield: openmm.app.ForceField,
        system_settings: config.SystemSettings,
    ):
        self.pressure = system_settings.pressure
        self.temperature = system_settings.temperature
        self.time_step = system_settings.time_step
        self.friction = 1 / picoseconds

        modeller = Modeller(molecule_file.topology, molecule_file.positions)
        modeller.addSolvent(forcefield, padding=1.1 * nanometer, model="tip4pew")
        self.topology = modeller.topology
        self.og_positions = modeller.positions

        small_molecule_atoms = mdtraj.Topology.from_openmm(modeller.topology).select(
            "resname UNL"
        )

        system = forcefield.createSystem(
            self.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=1.5 * amu,
        )
        logging.info(msg="Default box vectors")
        for axis in system.getDefaultPeriodicBoxVectors():
            logging.info(axis)

        factory = alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = alchemy.AlchemicalRegion(
            alchemical_atoms=small_molecule_atoms,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
            # default settings
            softcore_alpha=0.5,
            softcore_a=1.0,
            softcore_b=1.0,
            softcore_c=6.0,
            softcore_beta=0.0,  # 0.0 turns off softcore scaling of electrostatics (default)
        )
        self.NVT_alchemical_system = factory.create_alchemical_system(
            reference_system=system, alchemical_regions=alchemical_region
        )

        alchemical_state = alchemy.AlchemicalState.from_system(
            self.NVT_alchemical_system
        )
        composable_states = [alchemical_state]

        # compound alchemical states a and d into one state
        ts = openmmtools.states.ThermodynamicState(
            self.NVT_alchemical_system, self.temperature
        )
        self.NVT_compound_state = openmmtools.states.CompoundThermodynamicState(
            thermodynamic_state=ts, composable_states=composable_states
        )

        self.NPT_alchemical_system = copy.deepcopy(self.NVT_alchemical_system)
        self.NPT_compound_state = copy.deepcopy(self.NVT_compound_state)

        # add barostat and set presssure
        self.NPT_alchemical_system.addForce(
            MonteCarloBarostat(self.pressure, self.temperature, 25)
        )
        # Prime OpenMMtools to anticipate systems with barostats
        self.NPT_compound_state.pressure = self.pressure

    def build_simulation(
        self, system: openmm.System
    ) -> Tuple[openmm.app.Simulation, Type[openmm.openmm.Integrator]]:
        integrator = LangevinMiddleIntegrator(
            self.temperature, self.friction, self.time_step
        )
        platform = openmm.Platform.getPlatformByName("CUDA")
        platformProperties = {"CudaPrecision": "mixed"}
        simulation = openmm.app.Simulation(
            self.topology, system, integrator, platform, platformProperties
        )
        return simulation, integrator


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


def run_simulation(alchemical_system: AlchemicalSystem, lambda_scheme: LambdaScheme):
    nlambda = len(lambda_scheme)
    nstates = len(lambda_scheme)
    # u_kln = np.zeros([nstates, nstates, niter], np.float64)

    nvt_sim, nvt_integrator = alchemical_system.build_simulation(
        alchemical_system.NVT_alchemical_system
    )
    npt_sim, npt_integrator = alchemical_system.build_simulation(
        alchemical_system.NPT_alchemical_system
    )

    # Iterate over alchemical states
    for i, (sterics_lambda, electrostatic_lambda) in enumerate(lambda_scheme):
        # init position of atoms
        nvt_sim.context.setPositions(alchemical_system.og_positions)
        npt_sim.context.setPositions(alchemical_system.og_positions)


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

    forcefield = ForceField("tip4pew.xml")
    loaded_molecule = load_small_molecule(cfg.file_name, cfg.smiles, forcefield)

    alchemical_system = AlchemicalSystem(
        molecule_file=loaded_molecule, forcefield=forcefield, system_settings=cfg
    )

    lambda_scheme = LambdaScheme(cfg.sterics_lambdas, cfg.electrostatics_lambdas)
    run_simulation(alchemical_system, lambda_scheme)

    # simulation = Simulation(
    #     modeller.topology,
    #     alchemical_system,
    #     integrator,
    #     platform=Platform.getPlatformByName("CUDA"),
    #     platformProperties={"CudaPrecision": "mixed"},
    # )
    # simulation.context.setPositions(modeller.positions)
    # simulation.reporters.append(PDBReporter("output.pdb", 512))
    # simulation.reporters.append(
    #     StateDataReporter(
    #         "md_log.txt",
    #         512,
    #         step=True,
    #         potentialEnergy=True,
    #         temperature=True,
    #         volume=True,
    #     )
    # )
    # print("Minimizing energy")
    # simulation.minimizeEnergy()


if __name__ == "__main__":
    sys.exit(main())
