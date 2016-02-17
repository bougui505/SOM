#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-02-03 12:33:21 (UTC+0100)

from __future__ import print_function

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
import sys

class equilibration:
    """
    """
    def __init__(self, filename_pdb, forcefield_name='amber99sb.xml',
                 water_model_name='tip3p.xml'):
        self.filename_pdb = filename_pdb
        self.pdb = app.PDBFile(filename_pdb)
        self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = app.ForceField(forcefield_name, water_model_name)
        self.system = None
        self.integrator = None
        self.platform = None
        self.simulation = None

    def add_solvent(self, padding=1.0*unit.nanometers,
                    ionicStrength=0.1*unit.molar):
        print("Adding solvent...")
        self.modeller.addSolvent(self.forcefield, padding=padding,
                                 ionicStrength=ionicStrength)
        print("done")

    def create_system(self, nonbondedMethod=app.PME,
                    nonbondedCutoff=1.0*unit.nanometers,
                    constraints=app.AllBonds, rigidWater=True,
                    ewaldErrorTolerance=0.0005, temperature=300*unit.kelvin,
                    collision_rate=1.0/unit.picoseconds,
                    timestep=2.0*unit.femtoseconds, platform_name='OpenCL',
                    CpuThreads=1):
        print("Creating system...")
        self.system = self.forcefield.createSystem(self.modeller.topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
            constraints=constraints, rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance)
        self.integrator = mm.LangevinIntegrator(temperature, collision_rate,
                                                timestep)
        self.platform = mm.Platform.getPlatformByName(platform_name)
        if platform_name == 'CPU':
            self.platform.setPropertyDefaultValue('CpuThreads', '%s'%CpuThreads)
        self.simulation = app.Simulation(self.modeller.topology, self.system,
                                        self.integrator, self.platform)
        self.simulation.context.setPositions(self.modeller.positions)
        self.simulation.context.setVelocitiesToTemperature(temperature)
        print("done")

    def minimize(self):
        print("Minimizing...")
        self.simulation.minimizeEnergy()
        print("done")

    def equilibrate(self, number_of_steps=15000, report_interval=1000,
                    filename_output_pdb="equilibrated.pdb",
                    filename_output_log="openmm_equilibration.log"):
        print("Equilibrating...")
        self.simulation.reporters.append(app.StateDataReporter(filename_output_log,
            report_interval, step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True, temperature=True, volume=True,
            density=True, progress=True, remainingTime=True, speed=True,
            totalSteps=number_of_steps, separator='\t'))
        self.simulation.step(number_of_steps)
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(self.simulation.topology, positions,
                              open(filename_output_pdb, 'w'))
        print("done")

class production():
    """
    """
    def __init__(self, filename_pdb, forcefield_name='amber99sb.xml',
                 water_model_name='tip3p.xml'):
        self.filename_pdb = filename_pdb
        self.pdb = app.PDBFile(filename_pdb)
        self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = app.ForceField(forcefield_name, water_model_name)
        self.system = None
        self.integrator = None
        self.platform = None
        self.simulation = None

    def create_system(self, nonbondedMethod=app.PME,
                    nonbondedCutoff=1.0*unit.nanometers,
                    constraints=None, rigidWater=True,
                    ewaldErrorTolerance=0.0005, temperature=300*unit.kelvin,
                    collision_rate=1.0/unit.picoseconds,
                    timestep=1.0*unit.femtoseconds, platform_name='OpenCL',
                    CpuThreads=1):
        """
        Same function as in equilibration class except for the following default values:
        • constraints = None
        • timestep = 1fs instead of 2fs
        """
        print("Creating system...")
        self.system = self.forcefield.createSystem(self.modeller.topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
            constraints=constraints, rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance)
        self.integrator = mm.LangevinIntegrator(temperature, collision_rate,
                                                timestep)
        self.platform = mm.Platform.getPlatformByName(platform_name)
        if platform_name == 'CPU':
            self.platform.setPropertyDefaultValue('CpuThreads', '%s'%CpuThreads)
        self.simulation = app.Simulation(self.modeller.topology, self.system,
                                        self.integrator, self.platform)
        self.simulation.context.setPositions(self.modeller.positions)
        self.simulation.context.setVelocitiesToTemperature(temperature)
        print("done")

    def run(self, number_of_steps=1000, report_interval=10,
                    filename_output_dcd="trajectory.dcd",
                    filename_output_log="openmm_production.log",
                    filename_output_checkpoint="trajectory.chk"):
        """
        The checkpoint is created to restart a simulation.
        See https://goo.gl/G9mqzE for more details
        """
        print("MD production running...")
        self.simulation.reporters.append(app.DCDReporter(filename_output_dcd,
                                                         report_interval))
        self.simulation.reporters.append(app.StateDataReporter(filename_output_log,
            report_interval, step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True, temperature=True, volume=True,
            density=True, progress=True, remainingTime=True, speed=True,
            totalSteps=number_of_steps, separator='\t'))
        self.simulation.step(number_of_steps)
        with open(filename_output_checkpoint, 'wb') as f:
            f.write(self.simulation.context.createCheckpoint())
        print("done")

    def restart(self, filename_input_checkpoint="trajectory.chk",
                number_of_steps=1000, report_interval=10,
                filename_output_dcd="trajectory.dcd",
                filename_output_log="openmm_production.log",
                filename_output_checkpoint="trajectory.chk"):
        """
        Restart an MD from checkpoint file.
        See: https://goo.gl/G9mqzE for more details
        """
        # Load the checkpoint
        with open(filename_input_checkpoint, 'rb') as f:
            self.simulation.context.loadCheckpoint(f.read())
        self.run(number_of_steps=number_of_steps,
                 report_interval=report_interval,
                 filename_output_dcd=filename_output_dcd,
                 filename_output_log=filename_output_log,
                 filename_output_checkpoint=filename_output_checkpoint)
