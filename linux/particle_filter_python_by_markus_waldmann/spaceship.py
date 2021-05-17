# The spaceship class
#
# implements a simulated spaceship of
# the aliens that has the ability to
# split into several parts, where the
# parts can move independently from each other!
#
# ---
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
# ported from C++ to Python by Markus Waldmann.

from dataclasses import dataclass
import cv2 as cv
from params import *
import numpy as np


# Create Struct Point with x,y and operator overloading
@dataclass
class Point:
    x: float
    y: float

    # operator +=
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return Point(self.x, self.y)

    # operator -=
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return Point(self.x, self.y)

    # operator =
    def __copy__(self):
        return Point(self.x, self.y)  # copy of object


@dataclass
class PartInfos:
    location: Point
    moveVec: Point


@dataclass
class WorldSize:
    height: ()
    width: ()


class Spaceship:
    def __init__(self, nr_of_parts, world_size):
        # 1. store infos
        self.nr_of_parts = nr_of_parts
        self.world_size = WorldSize(world_size[0], world_size[1])
        self.part_infos = list()

        # 2. initialize all parts
        next_part_location = Point(self.world_size.width // 2, self.world_size.height // 2)
        rnd_common_move_vec = self.get_rnd_move_vec()
        for part_nr in range(nr_of_parts):
            # 2.1 create new part info object
            # 2.2 store pointer to that new part info object
            self.part_infos.append(PartInfos(next_part_location, rnd_common_move_vec))
            # 2.3 compute next part location
            next_part_location += Point(SPACESHIP_PART_SIZE_IN_PIXELS, 0)

    def move(self):
        # 1. move all parts according to their move vectors
        for part_info in self.part_infos:
            # 1.1 move part
            part_info.location += part_info.moveVec

            # 1.2 teleport?
            if SPACE_SHIPS_CAN_TELEPORT and (np.random.randint(0, 100) == 0):
                part_info.location.x = np.random.randint(0, self.world_size.width)
                part_info.location.y = np.random.randint(0, self.world_size.height)

            # 1.3 make sure, parts do not leave the 2D world
            if part_info.location.x - SPACESHIP_PART_SIZE_IN_PIXELS < 0:
                part_info.location.x = SPACESHIP_PART_SIZE_IN_PIXELS
                part_info.moveVec.x = +1
            if part_info.location.y - SPACESHIP_PART_SIZE_IN_PIXELS < 0:
                part_info.location.y = SPACESHIP_PART_SIZE_IN_PIXELS
                part_info.moveVec.y = +1
            if part_info.location.x + SPACESHIP_PART_SIZE_IN_PIXELS > self.world_size.width:
                part_info.location.x = self.world_size.width - SPACESHIP_PART_SIZE_IN_PIXELS
                part_info.moveVec.x = -1
            if part_info.location.y + SPACESHIP_PART_SIZE_IN_PIXELS > self.world_size.height:
                part_info.location.y = self.world_size.height - SPACESHIP_PART_SIZE_IN_PIXELS
                part_info.moveVec.y = -1

            # 1.4 compute new motion vector for this part?
            if np.random.randint(0, 250) == 0:
                part_info.moveVec = self.get_rnd_move_vec()

    def draw_yourself_into_this_image(self, image):
        # 1. draw each part individually
        for part_info in self.part_infos:
            # 1.1 draw filled box
            start_point = (int(part_info.location.x - SPACESHIP_PART_SIZE_IN_PIXELS // 2),
                           int(part_info.location.y - SPACESHIP_PART_SIZE_IN_PIXELS // 2))
            end_point = (int(part_info.location.x + SPACESHIP_PART_SIZE_IN_PIXELS // 2),
                         int(part_info.location.y + SPACESHIP_PART_SIZE_IN_PIXELS // 2))
            cv.rectangle(image, start_point, end_point, (0,255,0), -1)
            # 1.2 draw rectangle around box
            cv.rectangle(image, start_point, end_point, (0, 255, 200), 1)

    def get_part_info_vector(self):
        return self.part_infos

    def get_rnd_move_vec(self):
        rnd_x = np.random.randint(-1, 1)  # -1.0,1
        rnd_y = np.random.randint(-1, 1)  # -1.0,1
        return Point(rnd_x, rnd_y)
