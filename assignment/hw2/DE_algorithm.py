import numpy as np
from numpy import array
from numpy.random import rand
import random
from tqdm import tqdm
from utils import main_obj_function, Softmax

## Differential Evolution algorithm
class Diff_Evo(object):
    def __init__(self, pop_size=100, weight_F=1, cross_proby=0.5):
        self.pop_size = pop_size
        self.weight_F = weight_F
        self.cross_proby = cross_proby
        self.softmax = Softmax
        self.main_obj_function = main_obj_function()
    
    def forward(self, max_iteration=200):
        global_best_each_iter = []
        point_list = list(map(self.random_generate_sample, list(range(self.pop_size))))
        for t in tqdm(range(max_iteration)):
            if t % 2 == 0:
                function_value = list(map(self.main_obj_function.forward, point_list))

            new_point_list = list(map(self.get_new_point, list(range(self.pop_size)), [point_list]*self.pop_size))
            point_list = list(map(self.update_point, point_list, new_point_list))

            global_best_each_iter.append(max(list(map(self.main_obj_function.forward,point_list))))

        # return final_value
        final_value = list(map(self.main_obj_function.forward,point_list))
        final_value_index = final_value.index(max(final_value))

        # global_best_each_iter.append(final_value)
        return dict({'max_value':max(final_value), 'parameter':point_list[final_value_index], 'global_best_each_iter':global_best_each_iter})

    def random_generate_sample(self, *arg):
        random_x = np.sort(rand(3)) * 30
        random_w = self.softmax(rand(3))
        init_point = np.concatenate([random_x, random_w])
        return init_point

    def get_new_point(self, index, points, update_type='exp'):
        point_1, point_2, point_3 = random.sample(points[:index] + points[index+1:], 3)
        velocity = point_1 + self.weight_F * (point_2 - point_3)
        new_point = points[index].copy()
        min_value, max_value = sorted(random.sample(range(6),2))
        max_value += 1
        new_point[min_value:max_value] = velocity[min_value:max_value]
        for i in range(3):
            if new_point[i] < 0:
                new_point[i] = 0
            elif new_point[i] > 30:
                new_point[i] = 30
        new_point[:3] = np.sort(new_point[:3])
        new_point[3:] = self.softmax(new_point[3:])
        return new_point
    
    def update_point(self, point_1, point_2):
        function_value_1 = self.main_obj_function.forward(xs_and_ws=point_1)
        function_value_2 = self.main_obj_function.forward(xs_and_ws=point_2)
        final_point = point_1 if function_value_1 > function_value_2 else point_2
        return final_point