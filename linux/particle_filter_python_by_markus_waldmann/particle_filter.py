# A straightforward implementation
# of the particle filter idea
#
# A particle filter is a sample based approach
# for recursive Bayesian filtering
# The particles are a population based discrete
# representation of a probability density function.
#
# The filter recursively updates
#
#   - the particle locations according to a
#     probabilistic motion model
#     (prediction update step)
#
#   - recomputes importance weights for each particle
#     (measurement update step)
#
#   - resamples the particles according to the current
#     pdf represented by the importance weights
#
# ---
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
# ported from C++ to Python by Markus Waldmann.

from dataclasses import dataclass
import numpy as np
from params import *


#
# for each particle we store its location in state space
# and an importance weight
@dataclass
class Particle:
    state: np.ndarray
    weight: float

    # operator =
    def __copy__(self):
        return Particle(self.state, self.weight)  # copy of object


all_particles = list()  # list of all particles
particle_with_highest_weight = Particle(np.zeros(1), 0)
#
# base class for motion & measurement update models
#
class Particle_filter_update_model:
    def particle_filter_update_model(self):
        pass

    #
    # update location or important weight of the specified particle
    #
    def update_particle(self, particle):
        # default: no motion at all / no importance weight change
        # implement your motion model + perception model in an own subclass
        pass

#
# represents a probability distribution using
# a set of discrete particles
#
class Particle_filter:
    def __init__(self, population_size, state_space_dimension):
        self.ptr_user_data = None
        # 1. save infos about nr of particles to generate
        #    & dimension of the state space
        self.population_size = population_size
        self.state_space_dimension = state_space_dimension

        # 2. create the desired number of particles
        for i in range(population_size):
            # 2.1 create new particle object
            # 2.2 create vector for storing the particles location
            #     in state space
            state = np.zeros(state_space_dimension)
            # 2.3 set initial weight of this particle
            weight = 1.0/population_size
            self.particle = Particle(state,weight)
            # 2.4 store pointer to this particle
            all_particles.append(self.particle)

        # 3. prepare arrays for minimum and maximum coordinates
        #    of each state space dimension
        self.min_values = np.zeros(state_space_dimension)
        self.max_values = np.zeros(state_space_dimension)
        self.range_sizes = np.zeros(state_space_dimension)

        # 4. we have no motion and perception model yet
        self.your_prediction_model = None
        self.your_perception_model = None

        # 5. start locations of particles not yet set!
        self.start_locations_initalized = False

        # 6. no update steps done so far
        self.nr_update_steps_done = 0

        # 7. helper data structure for Monte Carlo step
        self.segmentation_of_unit_interval = np.zeros(population_size + 1)

    #
    # method to set pointer to user data
    # needed to access in motion or perception model
    #
    def set_userdata(self, ptr_user_data):
        self.ptr_user_data = ptr_user_data

    #
    # reset positions & weights of all particles
    # to start conditions
    #
    def reset_positions_and_weights(self):
        # 1. reset positions
        self.set_random_start_states_for_all_particles()

        # 2. reset weights
        for p in all_particles:
            p.weight = 1.0 / self.population_size

    #
    # should be used by the user to specify in which range [min_value,max_value]
    # the <param_nr>-th parameter of the state space lives
    #
    def set_param_ranges(self, param_nr, min_value, max_value):
        self.min_values[param_nr] = min_value
        self.max_values[param_nr] = max_value
        self.range_sizes[param_nr] = max_value - min_value

    #
    # for the specified particle we guess a random start location
    #
    def set_random_start_state_for_specified_particle(self, particle):
        for i in range(self.state_space_dimension):
            particle.state[i] = np.random.randint(self.min_values[i], self.max_values[i])

    #
    # set initial location in state space for all particles
    #
    def set_random_start_states_for_all_particles(self):
        for particle in all_particles:
            self.set_random_start_state_for_specified_particle(particle)
        self.start_locations_initalized = True

    #
    # returns a copy of an existing particle
    # the particle to be copied is chosen according to
    # a probability that is proportional to its importance weight
    #
    def sample_one_particle_according_to_importance_weight_pdf(self):
        # 1. guess a random number from [0,1]
        rndVal = np.random.uniform(0, 1)

        # 2. to which particle does the interval segment belong
        #    in which this random number lies?
        idx = -1
        for i, particle in enumerate(all_particles):
            # 2.1 get next segment of partition of unit interval
            a = self.segmentation_of_unit_interval[i]
            b = self.segmentation_of_unit_interval[i + 1]

            # 2.2 does the rndVal lie in the interval [a,b] of [0,1] that belongs to particle i?
            if a <= rndVal <= b:
                idx = i
                break

        if idx == -1:
            idx = len(all_particles) - 1

        # 3. particle with index <idx> has won! we will resample this particle for the next iteration!
        winner_particle = all_particles[idx]

        # 4. return a _similar_ 'copy' of that particle
        # 4.1 generate new copy particle
        # 4.2 copy location of that particle in state space
        copy_state = winner_particle.state.copy()

        # 4.3 copy shall be similar, not 100% identical
        value_range = 0.01 * self.range_sizes
        for state in copy_state:
            state += np.random.uniform(-value_range, value_range)

        # 4.4 weight is reset to 1/N
        copy_weight = 1.0 / self.population_size

        # 5. return particle copy
        return Particle(copy_state, copy_weight)

    #
    # one particle filter update step
    #
    def update(self):
        global all_particles
        # 1. did the user specify a motion and a perception update model?
        if self.your_prediction_model is None or self.your_perception_model is None:
            return

        # 2. set initial particle locations?
        if not self.start_locations_initalized:
            self.set_random_start_states_for_all_particles()

        # 3. update each particle
        # 3.1 get next particle
        for particle in all_particles:
            # 3.2 move that particle according to prediction
            if DO_PREDICTION_STEP:
                self.your_prediction_model.update_particle(particle)

            # 3.3 move that particle according to measurement
            if DO_MEASUREMENT_CORRECTION_STEP:
                self.your_perception_model.update_particle(particle)

            # 3.4 make sure, particles do not leave state space!
            for i, state in enumerate(particle.state):
                if state < self.min_values[i]: state = self.min_values[i]
                if state > self.max_values[i]: state = self.max_values[i]

        # 4. normalize importance weights

        # 4.1 compute sum of all weights
        sum_weights = 0
        for particle in all_particles:
            sum_weights += particle.weight

        # 4.2 normalize each particle weight
        for particle in all_particles:
            particle.weight /= sum_weights

        # 5. resample complete particle population based on
        #    current importance weights of current particles?
        if RESAMPLING:
            # 5.1 compute division of unit interval [0,1]
            #     such that each particle gets a piece of that interval
            #     where the length of the interval is proportional to its importance weight
            next_border = 0.0
            self.segmentation_of_unit_interval[0] = 0.0
            # get next particle
            for i, particle in enumerate(all_particles):
                # compute next border
                next_border += particle.weight

                # compute next border in unit interval
                self.segmentation_of_unit_interval[i + 1] = next_border

            # 5.2 generate new particle population
            new_population = list()
            for particle in all_particles:
                new_population.append(self.sample_one_particle_according_to_importance_weight_pdf())

            # 5.2.1 Set X Percentage of the particles randomly for recover
            if RESAMPLING_PERCENTAGE:
                n_rnd_particles = (self.population_size // 100) * int(RESAMPLING_PERCENTAGE)
                for particle in new_population[:n_rnd_particles]:
                    self.set_random_start_state_for_specified_particle(particle)

            # 5.3 delete old population
            all_particles.clear()

            # 5.4 set new sample population as current population
            all_particles = new_population.copy()

        # 6. find particle with highest weight / highest prob
        global particle_with_highest_weight
        for particle in all_particles:
            if particle_with_highest_weight.weight < particle.weight:
                particle_with_highest_weight = particle

        return all_particles




