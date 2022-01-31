
import numpy as np
from math import pi, sin, gamma
from numpy.random import rand, normal
from random import choices
from tqdm import tqdm
from utils import argmin

class flower_pollination(object):

    def __init__(self, function, pop_size=50, low_bound=-20, up_bound=20, dimension=10, p=0.8, alpha=0.9, **kwargs):

        self.pop_size = pop_size
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.dimension = dimension
        self.obj_f = function
        self.p = p
        self.alpha = alpha

    def forward(self, iteration_time, **kwargs):
        init_points_list = self.random_generate_sampe(size=self.pop_size)
        points_list = init_points_list[:]

        init_function_values_list = [
            self.obj_f(sub_point, **kwargs) for sub_point in init_points_list
        ]
        function_values = init_function_values_list[:]

        global_best_value_each_step = []
        global_best_point_each_step = []

        global_best_index = argmin(function_values)
        global_best_value_each_step.append(function_values[global_best_index])
        global_best_point_each_step.append(points_list[global_best_index])

        for t in tqdm(range(iteration_time)):
            for i in range(self.pop_size):
                if rand(1) < self.p:
                    new_point = points_list[i] + self.levy_fly() * (
                        global_best_point_each_step[-1] - points_list[i]
                    )
                else:
                    rand_index = choices(range(self.pop_size), k=2)
                    new_point = points_list[i] + rand(self.dimension, 1) * (
                        points_list[rand_index[0]] - points_list[rand_index[1]]
                    )

                temp_function_value = self.obj_f(new_point, **kwargs)
                if temp_function_value < function_values[i]:
                    function_values[i] = temp_function_value
                    points_list[i] = new_point

            global_best_index = argmin(function_values)
            global_best_value_each_step.append(function_values[global_best_index])
            global_best_point_each_step.append(points_list[global_best_index])

        return global_best_value_each_step, global_best_point_each_step

    def levy_fly(self, ):
        beta = 1.5
        sigma_u = (
            (gamma(1 + beta) * sin(0.5 * pi * beta))
            / (gamma(0.5 * (1 + beta)) * beta * (2 ** (0.5 * (beta - 1))))
        ) ** (1 / beta)
        sigma_v = 1
        step_part1 = normal(0, sigma_u, (self.dimension, 1))
        step_part2 = normal(0, sigma_v, (self.dimension, 1))
        step = step_part1 / (np.abs(step_part2) ** (1 / beta))
        return self.alpha * step

    def random_generate_sampe(self, size, *arg):
        init_points = (rand(self.dimension, size) - 0.5) * 40
        init_points = list(np.split(init_points, indices_or_sections=size, axis=1))
        return init_points