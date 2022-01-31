

import numpy as np
from numpy.random import rand, randint, normal
from math import pi, sin, gamma
from tqdm import tqdm
from utils import argmin

class cuckoo_search(object):
    def __init__(self, function, pop_size=50, low_bound=-20, up_bound=20, dimension=10, alpha=0.1, pa=0.25, **kwargs):

        self.pop_size = pop_size
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.dimension = dimension
        self.obj_f = function
        self.alpha = alpha
        self.pa = pa

    def forward(self, iteration_time, **kwargs):

        init_points_list = self.random_generate_sample(size=self.pop_size)
        points_list = init_points_list[:]

        function_values = [self.obj_f(sub_point, **kwargs) for sub_point in init_points_list]

        global_best_value_each_step = []
        global_best_point_each_step = []
        global_best_index = argmin(function_values)
        global_best_value_each_step.append(function_values[global_best_index])
        global_best_point_each_step.append(points_list[global_best_index])

        for t in tqdm(range(iteration_time)):
            for i in range(self.pop_size):
                temp_point = points_list[i] + self.levy_fly()
                temp_point = self.adj_to_domain(temp_point)
                val = randint(low=0, high=self.pop_size, size=(1,)).item()
                temp_value = self.obj_f(temp_point, **kwargs)
                if temp_value < function_values[val]:
                    points_list[val] = temp_point
                    function_values[val] = temp_value

            worse_nest_index = sorted(
                range(len(function_values)), key=lambda k: function_values[k]
            )[: int(self.pop_size * self.pa)]

            new_points_list = self.generate_new_point(points_list, worse_nest_index)
            for index, worse_index in enumerate(worse_nest_index):
                points_list[worse_index] = new_points_list[index]

            function_values = [self.obj_f(sub_point, **kwargs) for sub_point in points_list]

            min_index = argmin(function_values)
            global_best_value_each_step.append(function_values[min_index])
            global_best_point_each_step.append(points_list[min_index])

        return global_best_value_each_step, global_best_point_each_step

    def levy_fly(self):
        beta = 1.5
        sigma_u = (
            (gamma(1 + beta) * sin(0.5 * pi * beta))
            / (gamma(0.5 * (1 + beta)) * beta * (2 ** (0.5 * (beta - 1))))
        ) ** (1 / beta)
        sigma_v = 1
        step_part1 = normal(0, sigma_u, (self.dimension, ))
        step_part2 = normal(0, sigma_v, (self.dimension, ))
        step = step_part1 / (np.abs(step_part2) ** (1 / beta))
        return self.alpha * step
    
    def generate_new_point(self, point_list, worse_points_index):
        number_of_new_points = len(worse_points_index)
        new_points = []
        for k in range(number_of_new_points):
            val = randint(low=0, high=self.pop_size, size=(2,))
            random_v = self.pa - rand(self.dimension)
            random_v[random_v < 0] = 0

            new_points.append(
                point_list[worse_points_index[k]] + 
                self.alpha * 0.005 * random_v * (point_list[val[0].item()] - point_list[val[1].item()])
            )
        return new_points

    def random_generate_sample(self, size, *arg):
        init_points = (rand(self.dimension, size) - 0.5) * 40
        init_points = np.split(init_points, indices_or_sections=size, axis=1)
        init_points = list(map(lambda x:x.reshape(-1), init_points))
        return init_points
    
    def adj_to_domain(self, x):
        x[x < self.low_bound] = self.low_bound
        x[x > self.up_bound] = self.up_bound
        return x
        