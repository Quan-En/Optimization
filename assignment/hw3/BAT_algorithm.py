
import numpy as np
from numpy.random import rand, normal
from tqdm import tqdm
from utils import argmin

class bat_algorithm(object):
    def __init__(self, function, pop_size=50, low_bound=-20, up_bound=20, dimension=10,
                 fmin=0, fmax=1, A0=1, alpha=0.9, r0=1, r=0.1, sigma=0.5, **kwargs):

        self.pop_size = pop_size
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.dimension = dimension
        self.obj_f = function

        self.fmin = fmin
        self.fmax = fmax

        self.v0 = [np.zeros((dimension, 1)) for _ in range(pop_size)]
        self.A0 = A0
        self.alpha = alpha
        self.r0 = r0
        self.r = r
        self.sigma = sigma
        
    def forward(self, iteration_time, **kwargs):
        v = self.v0[:]
        A_t = self.A0

        init_points_list = self.random_generate_sampe(size=self.pop_size)
        points_list = init_points_list[:]
        
        function_values = [self.obj_f(sub_point, **kwargs) for sub_point in init_points_list]
        
        
        global_best_value_each_step = []
        global_best_point_each_step = []
        
        global_best_index = argmin(function_values)
        global_best_value_each_step.append(function_values[global_best_index])
        global_best_point_each_step.append(points_list[global_best_index])
        
        for t in tqdm(range(iteration_time)):
            if t > 0:
                A_t *= self.alpha
                r_t = self.r0 * (1 - np.exp(np.array([- self.r * t])))
            else:
                A_t = self.A0
                r_t = self.r0 * (1 - np.exp(np.array([- self.r * 1])))

            for bat_index in range(self.pop_size):
                new_point, v[bat_index] = self.generate_new_point(points_list[bat_index], v[bat_index], global_best_point_each_step[-1])
                if rand(1) > r_t:
                    new_point = global_best_point_each_step[-1] + normal(0, self.sigma, (self.dimension, 1))
                new_point = self.adj_to_domain(new_point)
                
                function_value = self.obj_f(new_point, **kwargs)
                
                if function_value < global_best_value_each_step[-1] and rand(1) < A_t:
                    function_values[bat_index] = function_value
                    points_list[bat_index] = new_point

            global_best_index = argmin(function_values)
            
            global_best_value_each_step.append(function_values[global_best_index])
            global_best_point_each_step.append(points_list[global_best_index])
        
        return global_best_value_each_step, global_best_point_each_step
 
    def generate_new_point(self, point, velocity, current_best_point):
        f = self.fmin + (self.fmax - self.fmin) * rand(self.dimension, 1)
        updated_velocity = velocity + f * (point - current_best_point)
        updated_point = point + updated_velocity
        return updated_point, updated_velocity

        
    def adj_to_domain(self, x):
        low_index = x[:,0] < self.low_bound
        x[low_index,0] = self.low_bound
        up_index = x[:,0] > self.up_bound
        x[up_index,0] = self.up_bound
        return x
    
    def random_generate_sampe(self, size, *arg):
        init_points = (rand(self.dimension, size) - 1) * 20
        init_points = list(np.split(init_points, indices_or_sections=size, axis=1))
        return init_points