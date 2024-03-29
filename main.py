from pathlib import PosixPath
from sys import stdout
from typing import Iterator, Type, TypeAlias
import argparse
import config
import copy
import logging
import sys
import time

from openff.toolkit import Molecule
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    NoCutoff,
    PDBFile,
    PDBReporter,
    PME,
    Simulation,
    StateDataReporter,
)
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
    Quantity,
    amu,
    angstrom,
    atmospheres,
    femtoseconds,
    kelvin,
    kilocalories_per_mole,
    kilojoules,
    kilojoules_per_mole,
    mole,
    nanometer,
    picoseconds,
)
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtools import alchemy
from pymbar import MBAR, timeseries
import mdtraj
import numpy as np
import numpy.typing as npt
import openmm
import openmmtools

from config import SystemSettings

# from lmlp import lMLP
from new_lmlp import lMLP

STEPS_PER_ITER: int = 1024
INITIAL_TEMPERATURE = 50

BOHR2ANGSTROM = 0.529177210903  # CODATA 2018
HARTREE2EV = 27.211386245988  # CODATA 2018
KJMOL2EV = 0.1 / 6.02214076 / 1.602176634  # CODATA 2018
MM_CHARGES = {
    "H": 0.417,
    "O": -0.834,
}


def dump_info(
    filename: str,
    energies,
    all_forces,
    all_positions,
    all_atoms,
    only_forces: bool,
):
    unit = kilojoules_per_mole / nanometer
    with open(f"{filename}", "w") as fd:
        for energy, forces, positions, atoms in zip(
            energies, all_forces, all_positions, all_atoms
        ):
            fd.write(f"potential energy: {energy}\n")
            for force, pos, atom in zip(forces, positions, atoms):
                if only_forces:
                    fd.write(
                        f"{force[0].value_in_unit(unit)} "
                        + f"{force[1].value_in_unit(unit)} "
                        + f"{force[2].value_in_unit(unit)}\n"
                    )
                else:
                    fd.write(
                        f"{atom.element.symbol} "
                        + f"{pos[0].value_in_unit(nanometer)} "
                        + f"{pos[1].value_in_unit(nanometer)} "
                        + f"{pos[2].value_in_unit(nanometer)} "
                        + f"{force[0].value_in_unit(unit)} "
                        + f"{force[1].value_in_unit(unit)} "
                        + f"{force[2].value_in_unit(unit)}\n"
                    )


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
        self.friction = 1.0 / picoseconds
        self.equili_steps = system_settings.equilibration_per_window
        self.niter = system_settings.sampling_per_window
        self.hydrogen_mass = 2.0 * amu
        self.nonbonded_cutoff = 1.9 * nanometer
        # self.nonbonded_method = NoCutoff
        self.nonbonded_method = PME
        self.switch_distance = 1.5 * nanometer

        modeller = Modeller(molecule_file.topology, molecule_file.positions)
        self.only_etoh_system = forcefield.createSystem(
            modeller.topology,
            # nonbondedMethod=self.nonbonded_method,
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=self.nonbonded_cutoff,
            switchDistance=self.switch_distance,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogen_mass,
        )
        self.only_etoh_topology = modeller.topology
        self.only_etoh_position = modeller.positions

        modeller.addSolvent(
            forcefield,
            # padding=2.1 * nanometer,
            padding=2.0 * nanometer,
            model="tip3p",
        )
        self.topology = modeller.topology
        self.og_positions = modeller.positions

        small_molecule_atoms = mdtraj.Topology.from_openmm(modeller.topology).select(
            "resname UNL"
        )

        system = forcefield.createSystem(
            self.topology,
            nonbondedMethod=self.nonbonded_method,
            nonbondedCutoff=self.nonbonded_cutoff,
            switchDistance=self.switch_distance,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogen_mass,
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
            # annihilate_electrostatics=False,
            # annihilate_sterics=False,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
            # default settings
            # softcore_alpha=0.5,
            # softcore_a=1.0,
            # softcore_b=1.0,
            # softcore_c=6.0,
            softcore_beta=1.0,  # 0.0 turns off softcore scaling of electrostatics (default)
        )
        self.only_etoh_NVT_alchemical_system = factory.create_alchemical_system(
            reference_system=self.only_etoh_system, alchemical_regions=alchemical_region
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

        self.total_n_atoms = self.topology.getNumAtoms()
        self.small_molecule_n_atoms = len(small_molecule_atoms)

    def build_simulation(
        self,
        system: openmm.System,
        topology: openmm.app.topology.Topology,
        random_seed: int = 42,
    ) -> tuple[openmm.app.Simulation, Type[openmm.openmm.Integrator]]:
        integrator = LangevinMiddleIntegrator(
            self.temperature, self.friction, self.time_step
        )
        integrator.setRandomNumberSeed(random_seed)
        platform = openmm.Platform.getPlatformByName("CUDA")
        platformProperties = {"CudaPrecision": "mixed"}
        simulation = openmm.app.Simulation(
            topology, system, integrator, platform, platformProperties
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
        system=alchemical_system.NVT_alchemical_system,
        topology=alchemical_system.topology,
    )
    # npt_sim, npt_integrator = alchemical_system.build_simulation(
    #     system=alchemical_system.NPT_alchemical_system,
    #     topology=alchemical_system.topology,
    # )

    only_etoh_sim, only_etoh = alchemical_system.build_simulation(
        system=alchemical_system.only_etoh_NVT_alchemical_system,
        topology=alchemical_system.only_etoh_topology,
    )
    only_etoh_sim.context.setPositions(alchemical_system.only_etoh_position)

    nvt_steps = int(alchemical_system.equili_steps * 0.01)
    npt_steps = int(alchemical_system.equili_steps)

    # initialize stuff for MLP
    n_atoms = alchemical_system.total_n_atoms
    n_atoms_sys = alchemical_system.small_molecule_n_atoms
    # generalization_setting_file = "../2from_marco/model/MLP-EtOH+H2O.ini"
    # double_lmlp = lMLP(generalization_setting_file=generalization_setting_file)

    generalization_setting_file = "../3from_marco/model/MLP-EtOH+H2O-Ewaldv2.ini"
    double_lmlp = lMLP(generalization_setting_file=generalization_setting_file)

    # generalization_setting_file = "../2from_marco/model/MLP-only_EtOH.ini"
    # only_etoh_lmlp = lMLP(generalization_setting_file=generalization_setting_file)
    # generalization_setting_file = "../2from_marco/model/MLP-only_EtOH+H2O.ini"
    # etoh_h2o_lmlp = lMLP(generalization_setting_file=generalization_setting_file)

    # Array with 1.0 for QM atoms and 2.0 for MM atoms
    atomic_classes = np.ones(n_atoms)
    atomic_classes[n_atoms_sys:] += 1.0

    # Array with atomic charges of MM atoms
    atomic_charges = np.zeros(n_atoms)
    elements = np.array([str(atom.element.symbol) for atom in nvt_sim.topology.atoms()])
    atomic_charges[n_atoms_sys:] = np.array(
        [MM_CHARGES[ele] for ele in elements[n_atoms_sys:]]
    )

    # Iterate over alchemical states
    for i, (steric_l_i, electrostatic_l_i) in enumerate(lambda_scheme):
        logger.debug("Sterics lambda i: %1.3f", steric_l_i)
        logger.debug("Electrostatics lambda i: %1.3f", electrostatic_l_i)
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
        logger.info(
            "Performing NVT equilibration for %s",
            nvt_steps * alchemical_system.time_step,
        )
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
        # nvt_sim.context.setPositions(pos)
        # nvt_sim.context.setVelocities(vel)

        logger.info(
            "Performing NPT equilibration for %s",
            npt_steps * alchemical_system.time_step,
        )
        tic = time.perf_counter()
        npt_sim.step(npt_steps)
        # nvt_sim.step(npt_steps)
        tock = time.perf_counter()
        logger.info("Took %1.4f seconds", tock - tic)
        # _remove_reporters(nvt_sim)
        # _remove_reporters(npt_sim)

        # Production
        # _add_reporter(npt_sim, el=electrostatic_l_i, sl=steric_l_i)
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
            # nvt_sim.step(STEPS_PER_ITER)

            for j, (steric_l_j, electrostatic_l_j) in enumerate(lambda_scheme):
                # logger.debug("Sterics lambda j: %1.3f", steric_l_j)
                # logger.debug("Electrostatics lambda j: %1.3f", electrostatic_l_j)
                _apply_context(
                    alchemical_system.NPT_compound_state,
                    # alchemical_system.NVT_compound_state,
                    npt_sim.context,
                    # nvt_sim.context,
                    steric_l_j,
                    electrostatic_l_j,
                )

                USE_MLP = True
                if (
                    (steric_l_j == 1.0 and electrostatic_l_j == 1.0)
                    or (steric_l_j == 0.0 and electrostatic_l_j == 0.0)
                ) and USE_MLP:
                    logger.debug(f"{USE_MLP = }")
                    logger.debug(f"{steric_l_j = }")
                    logger.debug(f"{electrostatic_l_j = }")
                    logger.debug("In MLP predictive stage - Interacting")
                    """
                    calculated as
                    E_full = E_only-H20_MM + E_interacting-EtOH_MLP
                    ^^^ neglects the interaction contribution from water -> EtOH

                    E_full: E of full system (water + EtOH), everything interacting, MM
                    E_EtOH_MLP: E of EtOH sterically interacting with H2O predicted by MLP
                    E_no-interaction: E of full system (water + EtOH), only steric interaction, MM
                    E_EtOH-vac: E of EtOH in vacuum, MM
                    E_EtOH-vac-MLP: E of EtOH in vacuum, MLP

                    for the interacting system
                    E_full = E_EtOH_MLP + E_no-interaction - E_EtOH-vac

                    for the noninteracting system
                    E_full = E_EtOH-vac-MLP + E_no-interaction - E_EtOH-vac
                    """
                    # state = npt_sim.context.getState(getEnergy=True, getPositions=True)
                    state = nvt_sim.context.getState(getEnergy=True, getPositions=True)
                    positions = state.getPositions(asNumpy=True).value_in_unit(
                        nanometer
                    )
                    logger.debug(
                        f"Raw predicted potential energy: {state.getPotentialEnergy()}"
                    )
                    positions = np.array(positions).astype(float) * 10.0 / BOHR2ANGSTROM
                    # E_EtOH_MLP, _ = double_lmlp.predict(
                    E_EtOH_MLP, _ = etoh_h2o_lmlp.predict(
                        elements,
                        positions,
                        atomic_classes=atomic_classes,
                        atomic_charges=atomic_charges,
                        calc_forces=False,
                    )
                    E_EtOH_MLP *= HARTREE2EV / KJMOL2EV
                    logger.debug(f"{E_EtOH_MLP = }")

                    # steric = steric_l_i
                    # steric = 1.0
                    steric = steric_l_j
                    electrostatic = 0.0
                    _apply_context(
                        # alchemical_system.NPT_compound_state,
                        alchemical_system.NVT_compound_state,
                        # npt_sim.context,
                        nvt_sim.context,
                        steric,
                        electrostatic,
                    )
                    # state = npt_sim.context.getState(getEnergy=True, getPositions=True)
                    state = nvt_sim.context.getState(getEnergy=True, getPositions=True)
                    E_no_interaction = state.getPotentialEnergy()
                    logger.debug(
                        f"{E_no_interaction.value_in_unit(kilojoules_per_mole) = }"
                    )
                    only_etoh_sim.context.setPositions(state.getPositions()[:9])
                    only_etoh_state = only_etoh_sim.context.getState(
                        getEnergy=True, getPositions=True, getForces=True
                    )
                    E_EtOH_vac = only_etoh_state.getPotentialEnergy()
                    logger.debug(f"{E_EtOH_vac.value_in_unit(kilojoules_per_mole) = }")

                    if steric_l_j == 1.0 and electrostatic_l_j == 1.0:
                        E_full = (
                            E_EtOH_MLP
                            + E_no_interaction.value_in_unit(kilojoules_per_mole)
                            - E_EtOH_vac.value_in_unit(kilojoules_per_mole)
                        )

                    if steric_l_j == 0.0 and electrostatic_l_j == 0.0:
                        # E_EtOH_vac_MLP, _ = double_lmlp.predict(
                        E_EtOH_vac_MLP, _ = only_etoh_lmlp.predict(
                            elements[:9],
                            positions[:9],
                            atomic_classes=atomic_classes[:9],
                            atomic_charges=atomic_charges[:9],
                            calc_forces=False,
                        )
                        E_EtOH_vac_MLP *= HARTREE2EV / KJMOL2EV
                        logger.debug(f"{E_EtOH_vac_MLP = }")
                        E_full = (
                            E_EtOH_vac_MLP
                            + E_no_interaction.value_in_unit(kilojoules_per_mole)
                            - E_EtOH_vac.value_in_unit(kilojoules_per_mole)
                        )
                    potential = E_full
                    potential = Quantity(potential, unit=kilojoules_per_mole)
                    potential /= AVOGADRO_CONSTANT_NA

                    logger.debug(f"Predicted potential energy: {E_full}")
                    # logger.debug(
                    #     f"Actual potential energy: {state.getPotentialEnergy()}"
                    # )
                    volume = state.getPeriodicBoxVolume()
                # if (steric_l_j == 0.0 and electrostatic_l_j == 0.0) and USE_MLP:
                #     """
                #     calculated as
                #
                #     E_full: E of full system (water + EtOH), everything interacting, MM
                #     E_EtOH_MLP: E of EtOH sterically interacting with H2O predicted by MLP
                #     E_no-interaction: E of full system (water + EtOH), only steric interaction, MM
                #     E_EtOH-vac: E of EtOH in vacuum, MM
                #
                #     E_full = E_EtOH-vac-MLP + E_no-interaction - E_EtOH-vac
                #     """
                else:
                    # otherwise, grab the normal energies
                    # state = npt_sim.context.getState(getEnergy=True)
                    state = nvt_sim.context.getState(getEnergy=True)
                    volume = state.getPeriodicBoxVolume()
                    logger.debug(
                        f"Non-MLP predicted potential energy: {state.getPotentialEnergy()}"
                    )
                    potential = state.getPotentialEnergy() / AVOGADRO_CONSTANT_NA

                u_kln[i, j, iteration] = beta * (
                    potential + alchemical_system.pressure * volume
                )

            # recover alchemical state
            _apply_context(
                alchemical_system.NPT_compound_state,
                # alchemical_system.NVT_compound_state,
                npt_sim.context,
                # nvt_sim.context,
                steric_l_i,
                electrostatic_l_i,
            )
        # _remove_reporters(npt_sim)
        _remove_reporters(nvt_sim)

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

    # Default, units of energy is kj/mol
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
