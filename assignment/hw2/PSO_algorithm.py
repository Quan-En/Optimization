import numpy as np
from numpy import array
from numpy.random import rand
import random
from tqdm import tqdm
from utils import main_obj_function, Softmax

## Particle Swarm Optimization
class PSO(object):
    def __init__(self, pop_size=50, alpha=2, beta=2, velocity_theta=0.7):
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.velocity_theta = velocity_theta
        self.softmax = Softmax
        self.main_obj_function = main_obj_function()

    def forward(self, max_iteration=200):
        global_best_each_iter = []
        
        point_list = list(map(self.random_generate_sample, list(range(self.pop_size))))
        x_list = [sub_data[0] for sub_data in point_list]
        v_list = [sub_data[1] for sub_data in point_list]
        
        local_best_point = x_list[:]
        function_value = list(map(self.main_obj_function.forward, x_list))
        global_best_value = max(function_value)
        function_value_index = function_value.index(global_best_value)
        global_best_point = x_list[function_value_index]
        
        for t in tqdm(range(max_iteration)):
            
            new_v_list = list(map(self.eval_velocity, v_list, x_list, local_best_point, [global_best_point]*self.pop_size))
            x_list = list(map(self.get_new_point, new_v_list, x_list))
            
            local_best_point = list(map(self.update_local_best_point, local_best_point,x_list))

            function_value = list(map(self.main_obj_function.forward, local_best_point))
            function_value_index = function_value.index(max(function_value))
            sectional_global_best_point = local_best_point[function_value_index]
            sectional_global_best_value = max(function_value)
            global_best_point = sectional_global_best_point if sectional_global_best_value > global_best_value else global_best_point
            global_best_value = sectional_global_best_value if sectional_global_best_value > global_best_value else global_best_value
            v_list = new_v_list[:]
            global_best_each_iter.append(global_best_value)
        
        return dict({'max_value':global_best_value, 'parameter':global_best_point, 'global_best_each_iter':global_best_each_iter})


    def random_generate_sample(self, *arg):
        random_x = np.sort(rand(3)) * 30
        random_w = self.softmax(rand(3))
        init_point = np.concatenate([random_x, random_w])
        init_volocity = array([0, 0, 0, 0, 0, 0])
        return init_point, init_volocity

    def eval_velocity(self, v_t, x_t, local_bast, global_bast):
        local_v = self.beta * rand(6) * (local_bast - x_t)
        global_v = self.alpha * rand(6) * (global_bast - x_t)
        v_t_plus_1 = self.velocity_theta * v_t + local_v + global_v
        return v_t_plus_1

    def get_new_point(self, v_t_plus_1, x_t):
        x_t_plus_1 = x_t + v_t_plus_1
        for i in range(3):
            if x_t_plus_1[i] < 0:
                x_t_plus_1[i] = 0
            elif x_t_plus_1[i] > 30:
                x_t_plus_1[i] = 30
        x_t_plus_1[:3] = np.sort(x_t_plus_1[:3])
        x_t_plus_1[3:] = self.softmax(x_t_plus_1[3:])
        return x_t_plus_1
    
    def update_local_best_point(self, point_1, point_2):
        function_value_1 = self.main_obj_function.forward(xs_and_ws=point_1)
        function_value_2 = self.main_obj_function.forward(xs_and_ws=point_2)
        final_point = point_1 if function_value_1 > function_value_2 else point_2
        return final_point