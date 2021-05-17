# Particle Filter Demo
#
# In this demo an alien invasion is
# simulated.
#
# Unfortunately, the aliens have the
# ability to split their spaceships
# into several parts and move these
# parts independently.
#
# So we have to be able to represent the
# location of the spaceship by some
# *multi-modal representation*!
#
# For this, a Kalman filter is not appropriate.
#
# Instead, we try to track their
# spaceships with the help of a particle filter,
# which has the ability to represent multi-modal
# probability distributions. Fortunately!
#
# ---
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
# ported from C++ to Python by Markus Waldmann.

import cv2 as cv
import math
import time
import numpy as np

from spaceship import *
from params import *
from particle_filter import *

measurements = []
max_euclidean_distance = 0


#
# here we make use of some "knowledge" about possible movements
# of the spaceship in order to
# predict the next particle location in state space
#
class Update_by_prediction_model(Particle_filter_update_model):
    def update_particle(self, particle):
        # make a prediction where the object will be next

        # get (x,y,vx,vy) state of this particle
        x = particle.state[0]
        y = particle.state[1]
        vx = particle.state[2]
        vy = particle.state[3]

        # prediction step
        x += vx
        y += vy

        if USE_NOISE_IN_PREDICTION_STEP:
            x += np.random.randint(-5.0, 5.0)
            y += np.random.randint(-5.0, 5.0)
            vx += np.random.uniform(-0.1, 0.1)  # [-0.1 1.1]
            vy += np.random.uniform(-0.1, 0.1)  # [-0.1 0.1]
            # vx += np.random.randint(-5.0, 5.0);
            # vy += np.random.randint(-5.0, 5.0);

            # add noise to predicted particle position &
            # store new particle position in state space
            particle.state[0] = x
            particle.state[1] = y
            particle.state[2] = vx
            particle.state[3] = vy


class Update_by_measurement_model(Particle_filter_update_model):
    def update_particle(self, particle):
        # get current position of particle
        x = particle.state[0]
        y = particle.state[1]
        vx = particle.state[2]
        vy = particle.state[3]

        # find out what the nearest measurement to
        # this particle is
        min_d = 0.0
        nearest_measurement = 0
        for i, measurement in enumerate(measurements):
            m_x = measurement[0]
            m_y = measurement[1]
            m_vx = measurement[2]
            m_vy = measurement[3]

            diff_x = x - m_x
            diff_y = y - m_y
            diff_vx = vx - m_vx
            diff_vy = vy - m_vy
            d = math.sqrt(diff_x * diff_x + diff_y * diff_y + diff_vx * diff_vx + diff_vy * diff_vy)

            if i == 0 or d < min_d:
                min_d = d
                nearest_measurement = i

        # move particle "a little bit" into the direction of the
        # nearest measurement
        fac = MOVE_TO_MEASUREMENT_SPEED
        x = x + fac * (measurements[nearest_measurement][0] - x)
        y = y + fac * (measurements[nearest_measurement][1] - y)
        vx = vx + fac * (measurements[nearest_measurement][2] - vx)
        vy = vy + fac * (measurements[nearest_measurement][3] - vy)

        if RESAMPLING:
            # compute new weight of particle
            particle.weight = max_euclidean_distance - min_d

        # store new particle position in state space
        particle.state[0] = x
        particle.state[1] = y
        particle.state[2] = vx
        particle.state[3] = vy

# maps a value in [0,1] to a color
# using a heat map coding:
# blue --> white --> yellow --> green --> red
def map_prob_to_color(prob):
    # 1. define heat map with 6 colors
    #    from cold to hot:
    #    white -> blue -> cyan -> green -> yellow -> red
    NUM_Colors = 6
    HeatMapColors = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]])

    # 2. make sure, prob is not larger than 1.0
    if prob > 1.0:
        prob = 1.0

    # 3. compute bottom and top color in heat map
    prob *= (NUM_Colors - 1)
    idx1 = max(0, int(math.floor(prob))) - 1
    idx2 = idx1 + 1
    fract = prob - float(idx1)  # where are we between the two colors?

    # 4. compute some intermediate color between HeatMapColors[idx2] and HeatMapColors[idx1]
    B = (HeatMapColors[idx2][0] - HeatMapColors[idx1][0]) * fract + HeatMapColors[idx1][0]
    G = (HeatMapColors[idx2][1] - HeatMapColors[idx1][1]) * fract + HeatMapColors[idx1][1]
    R = (HeatMapColors[idx2][2] - HeatMapColors[idx1][2]) * fract + HeatMapColors[idx1][2]

    # 5. set (B,G,R) values in a color / Vec3b object (Numpy)
    col = np.array([B, G, R])

    # 6. your heat map color is ready!
    return col


def get_image_of_continuous_probability_distribution(pf):
    start_time = time.process_time()

    sigma = 10.0  # 1,10,20
    normalization_fac = 1.0 / (math.sqrt(2 * math.pi) * sigma)

    # 1. prepare visualization image
    w = int(pf.max_values[0])
    h = int(pf.max_values[1])

    # 2. prepare a LUT for exp() function values?
    if USE_LUT_FOR_EXP:
        LUT_EXP = np.array([])
        len_of_diagonal = int(math.sqrt(w*w + h*h))
        for d in range(len_of_diagonal):
            LUT_EXP = np.append(LUT_EXP, [normalization_fac * np.exp(-(d*d) / (2.0*sigma*sigma))])

    print("generating image of continuous probability distribution:\n")

    # 3. prepare 2D array for storing probabilities
    probs = np.zeros((h, w))

    # 4. compute sum of all particle weights
    W = 0.0
    for p in all_particles:
        W += p.weight

    # 5. compute probability for each (x,y) position
    print("computing probability for each (x,y) position")
    max_prob = 0.0
    for y in range(0, h, SPEED_UP_KDE_STEPSIZE):
        print(".", end="")
        for x in range(0, w, SPEED_UP_KDE_STEPSIZE):
            # 5.1 compute probability P(x,y)
            # get the next particle
            prob = 0.0
            for particle in all_particles:
                # compute Eculidean distance between particle pos & (x,y) pos
                dx = particle.state[0] - x
                dy = particle.state[1] - y
                d = math.sqrt(dx*dx + dy*dy)

                # get Kernel value K(d)
                if not USE_LUT_FOR_EXP:
                    kernel_value = normalization_fac * np.exp(-(d*d) / (2.0*sigma*sigma))
                else:
                    kernel_value = LUT_EXP[int(d)]

                # update prob
                prob += particle.weight * kernel_value

            prob /= W

            # 5.2 later we need to know the max probability found
            #     in order to normalize the values for drawing a heat map
            if prob > max_prob:
                max_prob = prob

            # 5.3 store computed probability for that (x,y) position
            probs[y][x] = prob

    print("\n")
    print("max_prob = %.5f\n" % max_prob)

    mean = 0.0
    # 6. normalize all values to be in [0,1]
    for y in range(0, h, SPEED_UP_KDE_STEPSIZE):
        for x in range(0, w, SPEED_UP_KDE_STEPSIZE):
            probs[y][x] /= max_prob
            mean += probs[y][x]
    mean /= (h//SPEED_UP_KDE_STEPSIZE * w//SPEED_UP_KDE_STEPSIZE)
    print("%1.5f" % mean)

    # 7. generate image from 2D probability array
    image_prob_distri = np.zeros((h//SPEED_UP_KDE_STEPSIZE, w//SPEED_UP_KDE_STEPSIZE, 3))
    for y in range(h//SPEED_UP_KDE_STEPSIZE):
        for x in range(w//SPEED_UP_KDE_STEPSIZE):
            # 6.1 map that probability to a color
            col = map_prob_to_color(probs[y*SPEED_UP_KDE_STEPSIZE][x*SPEED_UP_KDE_STEPSIZE])

            # 6.2 set color in image
            image_prob_distri[y][x] = col

    elpased_time = time.process_time() - start_time
    print("Time elapsed for computing continuous density: %.2f sec" % elpased_time)

    return image_prob_distri


def cluster(particles):
    start_time = time.process_time()

    # here we cluster only in 2D image space (x,y),
    # although the particles "live" in a 4D state space (x,y,vx,vy)

    # 1. keep track of new clusters found
    clusters_found = list()

    # 2. do a mean shift trajectory for each particle
    counter_ms_updates = 0
    # 2.1 get the particle
    for p1 in particles:
        # 2.2 get 2D image position of the particle
        mean_pos = np.array([p1.state[0], p1.state[1]])

        # 2.3 now shift the mean position until it does not move
        #     any longer
        mean_shift_length = 2.0
        while mean_shift_length > CONVERGENCE_THRESHOLD_MEAN_SHIFT_VEC_LEN:
            new_mean_pos = np.zeros((2))
            norm_factor = 0.0
            # get the particle
            for p2 in particles:
                # get 2D position of particle p2
                p2_pos = np.array([p2.state[0], p2.state[1]])

                # compute Kernel distance
                n = (mean_pos - p2_pos).copy()
                distance = math.sqrt(n[0]*n[0]+n[1]*n[1])
                kernel_distance = 0.0
                if distance < CYLINDER_KERNEL_RADIUS:
                    kernel_distance = CYLINDER_KERNEL_RADIUS - distance

                # get particle weight
                w = p2.weight

                # compute contribution to sum
                contrib = kernel_distance * w * p2_pos

                # update sum
                new_mean_pos += contrib

                # update normalization factor
                norm_factor += kernel_distance * w

            new_mean_pos /= norm_factor

            # compute distance between old mean pos
            # and new mean pos
            n = (mean_pos - new_mean_pos).copy()
            mean_shift_length = math.sqrt(n[0] * n[0] + n[1] * n[1])

            # update mean position
            mean_pos = new_mean_pos

            counter_ms_updates += 1
        # 2.4 check whether the final mean position is similar
        #     to a cluster center already generated
        found = False
        for c in clusters_found:
            if np.linalg.norm(c - mean_pos) < TOLERANCE_CLUSTER_BUILDING:
                found = True
                break

        # 2.5 make up a new cluster?
        if not found:
            clusters_found.append(mean_pos)

    elapsed_time = time.process_time() - start_time

    print("I have found %d clusters (%d mean shift updates, %.5f sec).\n" %
          (len(clusters_found), counter_ms_updates, elapsed_time))

    return clusters_found

        # 1. load background image


#    image license is CC0 (public domain).
#    Download location:
#    https://pixabay.com/de/gewitter-wolken-horizont-brazilien-548951/
background_img_filename = "pic/earth_orbit_seen_from_space_pixabay_lowres.png"
background_image = cv.imread(background_img_filename)
# 2. image load error?

if background_image is None:
    print("Error! Could not read the image file: ", background_img_filename)

max_euclidean_distance = math.sqrt(background_image.shape[0] ** 2 + background_image.shape[1] ** 2)

# 3. make background image gray, but let it be a 3 color channel RGB image
#    so that we can draw colored information on it

background_image_gray = cv.cvtColor(background_image, cv.COLOR_RGB2GRAY)
background_image_gray = cv.cvtColor(background_image_gray, cv.COLOR_GRAY2RGB)

# 4. create an alien spaceship with NR_SPACESHIP_PARTS parts
#    and tell the spaceship how big our 2D world is

alien_spaceship = Spaceship(NR_SPACESHIP_PARTS, background_image.shape)

# 5. define the "noisy-ness" of our measurements &
#    prepare random generator according to the measurement
#    noise covariance matrix

R = np.array(([200.0, 0.0, 0.0, 0.0],
              [0.0, 200.0, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.0],
              [0.0, 0.0, 0.0, 0.1]))
mean_vec = np.array([0.0, 0.0, 0.0, 0.0])


def rnd_generator_measurement_noise(): return np.random.multivariate_normal(mean_vec, R)  # that's it in .py =)


# 6. prepare particle filter object
my_pf = None
my_update_by_prediction_model = Update_by_prediction_model()
my_update_by_measurement_model = Update_by_measurement_model()

# 7. the simulation loop:
simulation_step = 0
show_space_ship = False
key = -1
while True:
    # 7.1 clear screen & visualization image
    # print("Simulation step : ", simulation_step);

    # 7.2 copy background image into new visualization image
    # Mat image = background_image.clone();
    image = background_image.copy()

    # 7.3 move the alien spaceship
    alien_spaceship.move()

    # 7.4 simulate noisy measurements

    # 7.4.1 clear measurement vector
    measurements.clear()

    # 7.4.2 add a noisy measurement near to each part
    part_infos = alien_spaceship.get_part_info_vector()
    for part in part_infos:
        for i in range(NR_MEASUREMENTS_PER_SPACESHIP_PART):
            # prepare noise vector
            noise_4D_vec = rnd_generator_measurement_noise()

            # generate new measurement near to ground truth location & movement vector of that part
            measurement_4D_vec = np.array([part.location.x + noise_4D_vec[0],
                                           part.location.y + noise_4D_vec[1],
                                           part.moveVec.x + noise_4D_vec[2],
                                           part.moveVec.y + noise_4D_vec[3]])

            # add new measurement to list of measurements
            measurements.append(measurement_4D_vec)

    # 7.4.2 add some wrong measurements from time to time
    if SIMULATE_COMPLETELY_WRONG_MEASUREMENTS:
        always_generate_wrong_measurements = True
        if always_generate_wrong_measurements or np.random.randint(0, 1) == 0:
            for nr in range(NR_OF_COMPLETELY_WRONG_MEASUREMENTS):
                x = np.random.randint(0, background_image.shape[1])
                y = np.random.randint(0, background_image.shape[0])
                vx = np.random.randint(-1, 1)
                vy = np.random.randint(-1, 1)
                measurements.append(np.array([x, y, vx, vy]))

    # 7.4.3 test relocation speed of particles?
    # for testing the relocation of particles
    # using the RESAMPLING method
    # after measurements do not show up
    # any longer in some area of the state space

    if TEST_RELOCATION_OF_PARTICLES and simulation_step < 150:
        pass

    # 8. visualize all measurements by yellow circles
    for measurement in measurements:
        cv.circle(image, (int(measurement[0]), int(measurement[1])), 5, (0, 255, 255), -1)

    # 9. wait for user input
    if key == ord('x'):  # eXit particle filter demo
        exit()

    # 10. do one particle filter update step
    time_needed = 0
    if my_pf:
        start_time = time.process_time()
        all_particles = my_pf.update()
        time_needed = time.process_time() - start_time

    # 11. if the user wants to track the spaceship,
    # we initialize a particle filter
    if key == ord('t'):
        print("Starting tracking spaceship using particle filter...")

        # 11.1 if there is already a particle filter object, delete it
        if my_pf:
            all_particles.clear()
            del my_pf

        # 11.2 generate new particle filter object
        my_pf = Particle_filter(POPULATION_SIZE, 4)

        # 11.3 initialize particle filter:

        # 11.4 set state space dimensions
        my_pf.set_param_ranges(0, 0.0, float(image.shape[1]))  # x-position
        my_pf.set_param_ranges(1, 0.0, float(image.shape[0]))  # y-position
        my_pf.set_param_ranges(2, -1.0, 1.0)  # velocity x (vx)
        my_pf.set_param_ranges(3, -1.0, 1.0)  # velocity y (vy)

        # 11.5 set motion & measurement model
        my_pf.your_prediction_model = my_update_by_prediction_model
        my_pf.your_perception_model = my_update_by_measurement_model

        # 11.6 start with random positions in state space and uniform weight distribution
        my_pf.reset_positions_and_weights()

    # 12. visualize all particle locations
    if my_pf:
        # get the i-th particle
        for particle in all_particles:
            # get (x,y) position of particle
            x = particle.state[0]
            y = particle.state[1]

            # visualize particle position by a red circle
            cv.circle(image, (int(x), int(y)), 3, (0, 0, 255), 1)

    # 13. does the user want to generate a continous probability image based
    #     on the discrete particle positions?
    if key == ord('p'):
        if my_pf:
            cv.imshow("Continuous probability based on discrete particle positions",
                      get_image_of_continuous_probability_distribution(my_pf))


    # 15. user wants to turn on/off visualization of alien space ship
    if key == ord('s'):
        show_space_ship = not show_space_ship

    # 16. draw alien spaceship into image?
    if show_space_ship:
        alien_spaceship.draw_yourself_into_this_image(image)

    # 14. user wants to cluster the particle population
    if key == ord('c') or ALWAYS_CLUSTER_PARTICLES:
        if my_pf:
            # compute clusters
            clusters = cluster(all_particles)

            # show cluster centers as yellow circles
            for c in clusters:
                cv.circle(image, (int(c[0]), int(c[1])), SPACESHIP_PART_SIZE_IN_PIXELS // 2, (255, 0, 0), 2)

    # 17. show visualization image
    txt = "Step: " + str(simulation_step) + "(" + str(time_needed) + ")"
    cv.putText(image, txt, (10, image.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.imshow("Tracking an alien spaceship with a particle filter!", image)
    key = cv.waitKey(STEP_BY_STEP)

    # 18. time goes by...
    simulation_step += 1
