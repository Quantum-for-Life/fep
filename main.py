from pathlib import PosixPath
from typing import Tuple, Iterator, Type
import argparse
import config
import copy
import logging
import sys
from sys import stdout
import time

from pymbar import MBAR, timeseries
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
    kilojoules,
    mole,
)

from config import SystemSettings

STEPS_PER_ITER: int = 1024
INITIAL_TEMPERATURE = 50


def setup_logging(level: str) -> logging.Logger:
    """Setup basic logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # log to out
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=level)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)

    logger.propagate = False  # to stop duplicate log entries

    # # log to file
    # file_handler = logging.FileHandler("file.log")
    # file_handler.setFormatter(
    #     logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # )
    # logger.addHandler(file_handler)
    return logger


class CustomReporter(object):
    def __init__(self, file: str, reportInterval: int):
        self._out = open(file, "w")
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, True, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules / mole / nanometer)
        step_count = state.getStepCount()
        self._out.write(f"{step_count}\n")
        kinetic_energy = state.getKineticEnergy()
        self._out.write(f"kinetic energy: {kinetic_energy}\n")
        potential_energy = state.getPotentialEnergy()
        self._out.write(f"potential energy: {potential_energy}\n")
        for f in forces:
            self._out.write("%g %g %g\n" % (f[0], f[1], f[2]))


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
        self.logger = logging.getLogger(__name__)
        self.pressure = system_settings.pressure
        self.temperature = system_settings.temperature
        self.time_step = system_settings.time_step
        self.friction = 1 / picoseconds
        self.equili_steps = system_settings.equilibration_per_window
        self.niter = system_settings.sampling_per_window

        modeller = Modeller(molecule_file.topology, molecule_file.positions)
        modeller.addSolvent(
            forcefield,
            padding=1.5 * nanometer,
            model="tip3p",
        )
        self.topology = modeller.topology
        self.og_positions = modeller.positions

        small_molecule_atoms = mdtraj.Topology.from_openmm(modeller.topology).select(
            "resname UNL"
        )

        system = forcefield.createSystem(
            self.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.1 * nanometer,
            switchDistance=0.9 * nanometer,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=2.0 * amu,
        )

        # u, w, v = modeller.topology.getPeriodicBoxVectors()
        # system.setDefaultPeriodicBoxVectors(u, w, v)
        self.logger.info(msg="Default box vectors")
        for axis in system.getDefaultPeriodicBoxVectors():
            self.logger.info(axis)
        self.pbv = system.getDefaultPeriodicBoxVectors()

        factory = alchemy.AbsoluteAlchemicalFactory(
            alchemical_pme_treatment="direct-space"
        )
        alchemical_region = alchemy.AlchemicalRegion(
            alchemical_atoms=small_molecule_atoms,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
            # default settings
            softcore_alpha=0.5,
            softcore_a=1.0,
            softcore_b=1.0,
            softcore_c=6.0,
            softcore_beta=1.0,  # 0.0 turns off softcore scaling of electrostatics (default)
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
    logger = logging.getLogger(__name__)
    logger.debug(
        "Available GAFF force fields: %s", GAFFTemplateGenerator.INSTALLED_FORCEFIELDS
    )

    try:
        if file_name.suffix == ".pdb":
            loaded_molecule = PDBFile(f"{file_name}")
        else:
            raise ValueError(f"file format not implemented: {file_name}")
    except ValueError as e:
        logger.exception("ValueError")
        raise

    # molecule = Molecule.from_smiles(f"{smiles}")
    molecule = Molecule.from_file("ethanol.mol")
    molecule.assign_partial_charges("am1bcc")
    gaff = GAFFTemplateGenerator(molecules=molecule)
    logger.info("Using the following GAFF force field: %s", gaff.forcefield)
    forcefield.registerTemplateGenerator(gaff.generator)
    logger.info("Partial charges of molecule: %s", molecule.partial_charges)

    return loaded_molecule


def _apply_context(
    compound_state: openmmtools.states.CompoundThermodynamicState,
    context: openmm.Context,
    steric: float,
    electrostatic: float,
) -> None:
    compound_state.lambda_sterics = steric
    compound_state.lambda_electrostatics = electrostatic
    compound_state.apply_to_context(context)


def _add_reporter(simulation: openmm.app.Simulation, el: float, sl: float):
    simulation.reporters.append(
        openmm.app.StateDataReporter(
            stdout,
            reportInterval=5000,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            volume=True,
            temperature=True,
            # progress=True,
            # remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    simulation.reporters.append(PDBReporter(f"output_el_{el}_sl_{sl}.pdb", 100))
    simulation.reporters.append(CustomReporter(f"forces_el_{el}_sl_{sl}.txt", 100))


def _remove_reporters(simulation: openmm.app.Simulation) -> None:
    del simulation.reporters
    simulation.reporters = []


def run_simulation(alchemical_system: AlchemicalSystem, lambda_scheme: LambdaScheme):
    logger = logging.getLogger(__name__)
    nlambda = len(lambda_scheme)
    nstates = len(lambda_scheme)
    u_kln = np.zeros([nstates, nstates, alchemical_system.niter], np.float64)
    beta = 1.0 / (BOLTZMANN_CONSTANT_kB * alchemical_system.temperature)

    nvt_sim, nvt_integrator = alchemical_system.build_simulation(
        alchemical_system.NVT_alchemical_system
    )
    npt_sim, npt_integrator = alchemical_system.build_simulation(
        alchemical_system.NPT_alchemical_system
    )

    nvt_steps = int(alchemical_system.equili_steps * 0.01)
    npt_steps = int(alchemical_system.equili_steps)

    # Iterate over alchemical states
    for i, (steric_l_i, electrostatic_l_i) in enumerate(lambda_scheme):
        logger.debug("Sterics lambda i: %1.3f", steric_l_i)
        logger.debug("Electrostatics lambda i: %1.3f", electrostatic_l_i)
        # init position of atoms
        nvt_sim.context.setPositions(alchemical_system.og_positions)
        npt_sim.context.setPositions(alchemical_system.og_positions)

        # init lambda state
        _apply_context(
            alchemical_system.NVT_compound_state,
            nvt_sim.context,
            steric_l_i,
            electrostatic_l_i,
        )
        _apply_context(
            alchemical_system.NPT_compound_state,
            npt_sim.context,
            steric_l_i,
            electrostatic_l_i,
        )

        # init size of box, NVT never changes so dont need to init every state
        npt_sim.context.setPeriodicBoxVectors(*alchemical_system.pbv)

        # _add_reporter(nvt_sim, el=-0.0, sl=-0.0)
        # _add_reporter(npt_sim, el=-0.0, sl=-0.0)

        # preproduction: minimize and equilibriate
        logger.info("Minimizing")
        openmm.LocalEnergyMinimizer.minimize(nvt_sim.context)

        # NVT
        logger.info("Performing NVT equilibration for %s", nvt_steps * 2 * femtoseconds)
        tic = time.perf_counter()

        # NVT warming
        final_temperature = nvt_integrator.getTemperature() / kelvin
        nvt_sim.context.setVelocitiesToTemperature(INITIAL_TEMPERATURE)
        temps = np.linspace(INITIAL_TEMPERATURE, final_temperature, 10)
        logger.info(
            "Warming NVT from %1.4fk to %1.4fk", INITIAL_TEMPERATURE, final_temperature
        )
        for temp in temps:
            nvt_integrator.setTemperature(temp)
            nvt_integrator.step(int(nvt_steps / len(temps)))
        tock = time.perf_counter()
        logger.info("Took %1.4f seconds", tock - tic)

        # transfer positions and velocities to NPT system
        pos_vel = nvt_sim.context.getState(getPositions=True, getVelocities=True)
        pos, vel = pos_vel.getPositions(), pos_vel.getVelocities()

        # NPT
        # Set equilibriated pos and vel in NPT context
        npt_sim.context.setPositions(pos)
        npt_sim.context.setVelocities(vel)

        logger.info("Performing NPT equilibration for %s", npt_steps * 2 * femtoseconds)
        tic = time.perf_counter()
        npt_sim.step(npt_steps)
        tock = time.perf_counter()
        logger.info("Took %1.4f seconds", tock - tic)
        _remove_reporters(nvt_sim)
        _remove_reporters(npt_sim)

        # Production
        # kT = (
        #     AVOGADRO_CONSTANT_NA
        #     * BOLTZMANN_CONSTANT_kB
        #     * npt_integrator.getTemperature()
        # )
        _add_reporter(npt_sim, el=electrostatic_l_i, sl=steric_l_i)
        for iteration in range(alchemical_system.niter):
            logger.info(
                "Propagating iteration %d/%d in window %d/%d",
                iteration + 1,
                alchemical_system.niter,
                i + 1,
                len(lambda_scheme),
            )
            # propagate system in current state
            npt_sim.step(STEPS_PER_ITER)

            for j, (steric_l_j, electrostatic_l_j) in enumerate(lambda_scheme):
                logger.debug("Sterics lambda j: %1.3f", steric_l_j)
                logger.debug("Electrostatics lambda j: %1.3f", electrostatic_l_j)
                _apply_context(
                    alchemical_system.NPT_compound_state,
                    npt_sim.context,
                    steric_l_j,
                    electrostatic_l_j,
                )
                logger.debug(
                    "context electrostatics lambda: %1.4f",
                    alchemical_system.NPT_compound_state.lambda_electrostatics,
                )
                state = npt_sim.context.getState(getEnergy=True)
                volume = state.getPeriodicBoxVolume()
                logger.debug("Volume: %s", volume)
                potential = state.getPotentialEnergy() / AVOGADRO_CONSTANT_NA
                logger.debug(
                    "beta * (potential + alchemical_system.pressure * volume): %s",
                    beta * (potential + alchemical_system.pressure * volume),
                )
                u_kln[i, j, iteration] = beta * (
                    potential + alchemical_system.pressure * volume
                )
                # u_kln[i, j, iteration] = state.getPotentialEnergy() / kT

            # recover alchemical state
            _apply_context(
                alchemical_system.NPT_compound_state,
                npt_sim.context,
                steric_l_i,
                electrostatic_l_i,
            )
        _remove_reporters(npt_sim)

    # Subsample data to extract uncorrelated equilibrium timeseries
    N_k = np.zeros([nstates], np.int32)  # number of uncorrelated samples
    for k in range(nstates):
        [nequil, g, Neff_max] = timeseries.detect_equilibration(u_kln[k, k, :])
        indices = timeseries.subsample_correlated_data(u_kln[k, k, :], g=g)
        N_k[k] = len(indices)
        u_kln[k, :, 0 : N_k[k]] = u_kln[k, :, indices].T

    # Compute free energy differences
    mbar = MBAR(u_kln, N_k)

    # computing uncertainties may fail with an error for
    # pymbar versions > 3.0.3. See this issue: https://github.com/choderalab/pymbar/issues/419
    results = mbar.compute_free_energy_differences(compute_uncertainty=True)

    logger.info(
        "Free energy change to insert a particle = %1.16f",
        results["Delta_f"][nstates - 1, 0],
    )
    logger.info("Statistical uncertainty = %1.16f", results["dDelta_f"][nstates - 1, 0])


def main():
    parser = argparse.ArgumentParser(description="Calculate FEP")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        type=str,
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )
    args = parser.parse_args()
    logger = setup_logging(level=args.loglevel.upper())

    cfg = config.load_config(args.config)
    logger.info(f"Configuration loaded and validated: {cfg}")

    forcefield = ForceField("tip3p.xml")
    loaded_molecule = load_small_molecule(cfg.file_name, cfg.smiles, forcefield)

    alchemical_system = AlchemicalSystem(
        molecule_file=loaded_molecule, forcefield=forcefield, system_settings=cfg
    )

    lambda_scheme = LambdaScheme(cfg.sterics_lambdas, cfg.electrostatics_lambdas)
    logger.debug(
        "Lambda Scheme: {}".format(" \n".join(map(str, lambda_scheme._lambda_scheme)))
    )

    logger.info("Beginning simulation")
    run_simulation(alchemical_system, lambda_scheme)


if __name__ == "__main__":
    sys.exit(main())
