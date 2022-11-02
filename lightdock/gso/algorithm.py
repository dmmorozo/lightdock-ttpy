"""The Glowworm Swarm Optimization (GSO) algorithm implementation.

Algorithm is described in:
KRISHNANAND, K.N. AND GHOSE, D. 2009.
Glowworm swarm optimization for simultaneous capture of multiple local
optima of multimodal functions. Swarm Intelligence, 3, 2, June 2009,
87-124.
"""

import os
import math
import numpy as np 
import scipy.optimize

import tt
from tt.optimize import tt_min

from lightdock.constants import (
    DEFAULT_POSITIONS_FOLDER,
    DEFAULT_SWARM_FOLDER,
    DEFAULT_LIST_EXTENSION,
    DEFAULT_LIGHTDOCK_PREFIX,
    DEFAULT_NMODES_REC,
    DEFAULT_NMODES_LIG,
    MIN_EXTENT,
    MAX_EXTENT,
    DEFAULT_SETUP_FILE,
    DEFAULT_LIGHTDOCK_INFO,
    DEFAULT_STARTING_PREFIX,
    MAX_TRANSLATION,
    MAX_ROTATION,
    DEFAULT_SWARM_RADIUS,
    DEFAULT_MASK_FILE,
    DEFAULT_SWARM_DISTANCE,
    DEFAULT_SWARMS_PER_RESTRAINT,
)
from lightdock.gso.initializer import (
    RandomInitializer,
    FromFileInitializer,
    LightdockFromFileInitializer,
)
from lightdock.mathutil.ellipsoid import MinimumVolumeEllipsoid

from lightdock.util.logger import LoggingManager

log = LoggingManager.get_logger("lightdock")

class GSO(object):
    """GSO is the main simulation object"""

    def __init__(
        self,
        swarm,
        gso_parameters,
        random_number_generator,
        initial_coordinates_file="",
        local_minimization=False,
    ):
        self.swarm = swarm
        self.parameters = gso_parameters
        self.random_number_generator = random_number_generator
        self.initial_coordinates_file = initial_coordinates_file
        self.local_minimization = local_minimization
        self.n_calc = 0

    def target_func(self, x):
        """if x.ndim > 1:
            scoring = np.zeros(len(x))
            for i in range(len(x)):
                self.swarm.glowworms[0].landscape_positions[0].update_landscape_position(x[i])
                scoring[i] = -1.0 * self.swarm.glowworms[0].landscape_positions[0].evaluate_objective_function()
                self.n_calc = self.n_calc + 1
        else:"""
        self.swarm.glowworms[0].landscape_positions[0].update_landscape_position(x)
        scoring = -1.0 * self.swarm.glowworms[0].landscape_positions[0].evaluate_objective_function()
        self.n_calc = self.n_calc + 1
            
        """print(f'Step [{self.n_calc}] Energy = {scoring}')"""
        #print(scoring)
        return scoring

    def local_opt(self, x):

        self.swarm.glowworms[0].landscape_positions[0].update_landscape_position(x)
        score, x_min = self.swarm.glowworms[0].landscape_positions[0].minimize()
        return (-1.0*score, x_min)

    def run(
        self,
        simulation_steps,
        cluster_id=None,
        verbose=False,
        saving_path=".",
        save_intermediary=False,
        save_all_intermediary=False,
    ):
        # Calculate boundary for translations as maximum ligand radii + 1 nm
        lig_radii = np.amax(MinimumVolumeEllipsoid(self.swarm.glowworms[0].landscape_positions[0].ligand.coordinates[0].coordinates).radii)

        # Init max and min boundaries arrays
        max_boundary = np.array([ lig_radii + 10.0, 
                                  lig_radii + 10.0,
                                  lig_radii + 10.0,
                                  MAX_ROTATION,
                                  MAX_ROTATION, 
                                  MAX_ROTATION, 
                                  MAX_ROTATION])
        
        min_boundary = -max_boundary

        # Add possible normal modes of receptor and ligand
        if self.swarm.glowworms[0].landscape_positions[0].num_rec_nmodes > 0:
            for _ in range(self.swarm.glowworms[0].landscape_positions[0].num_rec_nmodes): 
                max_boundary.append(MAX_EXTENT)
                min_boundary.append(MIN_EXTENT)

        if self.swarm.glowworms[0].landscape_positions[0].num_lig_nmodes > 0:
            for _ in range(self.swarm.glowworms[0].landscape_positions[0].num_lig_nmodes): 
                max_boundary.append(MAX_EXTENT)
                min_boundary.append(MIN_EXTENT)

        # Number of dimensions
        ndim = min_boundary.size
        # Grid size for each dimension
        n0=self.parameters.ngrid
        # Max Rank of the TT
        rmax=self.parameters.rmax
        # Number of itarations
        nswp=simulation_steps

        log.info("TTPY Minimization parameters: ")
        log.info(f"Dimensions: {ndim}")
        log.info(f"Grid size: {n0}")
        log.info(f"Max rank: {rmax}")
        log.info(f"Number of iterations: {nswp}")
        log.info(f"Maximum ligand translation: {max_boundary[0]}")
        
        # Initialize random seed to be sure that they will differ between all simulations
        np.random.seed()

        # Call ttpy minimization
        score, x_min = tt_min.min_func(self.target_func, bounds_min=min_boundary, bounds_max=max_boundary, d=ndim, n0=n0, rmax=rmax, nswp=nswp)
        log.info(f"Total number of function evaluations: {self.n_calc}")

        # Save result
        self.swarm.glowworms[0].landscape_positions[0].update_landscape_position(x_min)
        self.swarm.glowworms[0].scoring = -1.0 * score
        self.swarm.save(100, saving_path)

    def report(self, output_file_name=""):
        """Writes to output_file_name if defined or to standard output the result of a GSO execution."""
        output = "GSO Execution Report:%s%s" % (os.linesep, os.linesep)
        output += "Seed: %s%s" % (self.random_number_generator.seed, os.linesep)
        output += "Number of Glowworms: %s%s" % (self.swarm.get_size(), os.linesep)
        if self.initial_coordinates_file:
            output += "Coordinates file: %s%s" % (
                self.initial_coordinates_file,
                os.linesep,
            )
        output += os.linesep
        output += "Algorithm parameters:%s" % os.linesep
        output += "Rho: %s%s" % (self.parameters.rho, os.linesep)
        output += "Gamma: %s%s" % (self.parameters.gamma, os.linesep)
        output += "Beta: %s%s" % (self.parameters.beta, os.linesep)
        output += "Initial Luciferin: %s%s" % (
            self.parameters.initial_luciferin,
            os.linesep,
        )
        output += "Initial Vision Range: %s%s" % (
            self.parameters.initial_vision_range,
            os.linesep,
        )
        output += "Maximum Vision Range: %s%s" % (
            self.parameters.max_vision_range,
            os.linesep,
        )
        output += "Maximum Number of Neighbors: %s%s" % (
            self.parameters.max_neighbors,
            os.linesep,
        )

        if output_file_name:
            output_file = open(output_file_name, "w")
            output_file.write(output)
            output_file.close()
        else:
            return output

    def __repr__(self):
        """String representation of the simulation"""
        return self.report()


class GSOBuilder(object):
    """Builds a generic GSO simulation object.

    The GSO object can be created using random initial positions for the
    glowworms (create function) or using initial positions read from a given file
    (create_from_file function).
    """

    def __init__(self):
        self._initializer = None

    def create(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        objective_function,
        bounding_box,
    ):
        """Creates a new GSO instance of the algorithm"""
        self._initializer = RandomInitializer(
            [objective_function],
            number_of_glowworms,
            gso_parameters,
            bounding_box,
            random_number_generator,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
        )

    def create_from_file(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        objective_function,
        bounding_box,
        initial_population_file,
    ):
        """Creates a new GSO instance of the algorithm reading the initial position of the glowworms
        agents from initial_population_file.
        """
        self._initializer = FromFileInitializer(
            [objective_function],
            number_of_glowworms,
            gso_parameters,
            bounding_box.dimension,
            initial_population_file,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
            initial_population_file,
        )


class LightdockGSOBuilder(object):
    """Creates a GSO simulation object for the LightDock framework"""

    def create_from_file(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        adapters,
        scoring_functions,
        bounding_box,
        initial_population_file,
        step_translation,
        step_rotation,
        step_nmodes,
        local_minimization,
        anm_rec,
        anm_lig,
    ):
        """Creates a new GSO instance of the algorithm reading the initial position of the glowworms
        agents from initial_population_file and using the scoring function adapter.
        """
        self._initializer = LightdockFromFileInitializer(
            adapters,
            scoring_functions,
            number_of_glowworms,
            gso_parameters,
            bounding_box.dimension,
            initial_population_file,
            step_translation,
            step_rotation,
            random_number_generator,
            step_nmodes,
            anm_rec,
            anm_lig,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
            local_minimization=local_minimization,
        )
