import numpy as np
from numpy import array
from numpy.random import rand
import random
from tqdm import tqdm
from utils import main_obj_function, Softmax


## Firefly algorithm
class FA(object):
    def __init__(self, pop_size=30, alpha=1, beta=1, gamma=0.1, theta=0.95):
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.softmax = Softmax
        self.main_obj_function = main_obj_function()

    def forward(self, max_iteration=100):
        all_point_list = []
        global_best_each_iter = []

        range_index = list(range(self.pop_size))
        init_point_list = list(map(self.random_generate_sample, range_index))
        all_point_list.append(init_point_list)
        
        function_values = list(map(self.main_obj_function.forward, init_point_list))
        global_best_value = max(function_values)
        
        for t in tqdm(range(max_iteration)):
            
            if 'temp_point_list' not in locals():
                temp_point_list = init_point_list[:]
                
            for i in range_index:
                inner_index = range_index[:]
                inner_index.remove(i)

                for j in inner_index:
                    if function_values[i] < function_values[j]:
                        
                        attractive = self.eval_attractive(self.eval_distence(temp_point_list[i], temp_point_list[j]))
                        velocity = attractive * (temp_point_list[j] - temp_point_list[i])
                        x_update = temp_point_list[i] + velocity + self.alpha * (self.theta ** t) * (2 * rand(6) - 1)
                        for k in range(3):
                            if x_update[k] < 0:
                                x_update[k] = 0
                            elif x_update[k] > 30:
                                x_update[k] = 30
                        x_update[:3] = np.sort(x_update[:3])
                        x_update[3:] = self.softmax(x_update[3:])
                    else:
                        x_update = temp_point_list[i]
                        
                    function_value = self.main_obj_function.forward(x_update)
                    # function_values[i] = function_value
                    temp_point_list[i] = x_update
                    
            function_values = list(map(self.main_obj_function.forward, temp_point_list))
            global_best_value = max(function_values) # global_best_value if global_best_value > function_value else function_value
            global_best_each_iter.append(global_best_value)
            # all_point_list.append(temp_point_list)
                        
        global_input_index = function_values.index(global_best_value)
        # return dict({})global_best_value, temp_point_list[global_input_index], all_point_list
        return dict({'max_value':global_best_value, 'parameter':temp_point_list[global_input_index], 'global_best_each_iter':global_best_each_iter})

    def random_generate_sample(self, *arg):
        random_x = np.sort(rand(3)) * 30
        random_w = self.softmax(rand(3))
        init_point = np.concatenate([random_x, random_w])
        return init_point
    
    def eval_attractive(self, distence):
        return self.beta / (1 + self.gamma * (distence)**2)
        
    def eval_distence(self, point1, point2):
        diff = (point1 - point2)
        distence = np.sqrt(np.sum(diff ** 2))
        return distence.item()