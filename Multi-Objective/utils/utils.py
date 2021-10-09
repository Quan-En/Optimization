"""
Several functions
version: 2021-09-12

- function (1): non_dominated_relation(x:torch.Tensor, target_set:list) -> bool
 -- usg: True = non-dominated all target points; False = dominated some target point
 --- warning: target_set should not contain repeat cases

- function (2): dominate_relation(x:torch.Tensor, target_set:list) -> bool
 -- usg: True = dominate some point in target set; False = nondominate all points in target set

- function (3): collect_efficient_solutions(candidate_set:list) -> list
- function (4): get_ref_point(pareto_set: list) -> tensor(m*1)

- function (5): calculate_volume(lower_corner_value: torch.Tensor, upper_corner_value: torch.Tensor) -> tensor(1, )

- function (6): plot_pareto_set_2D(pareto_set:list) -> plot

- function (7): RAAE(true_value, distribution_information:list) -> float

- function (8): print_model_parameters(model) -> null
"""

import torch
import numpy as np

import timeit
from paretoset import paretoset
from collections import namedtuple

from matplotlib import pyplot as plt
#from utils.AVLTree import non_dominated, AVLTree#  as my_avl_tree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

def non_dominated(x, y):
    result = not all(x >= y)
    return result



# Target set should equal to [Tensor(m*1), ..., Tensor(m*1)]
# Target set should not contain repeat cases
# return: True = non-dominated all target points; False = dominated some target point
def non_dominated_relation(x:torch.Tensor, target_set:list):
    for target_point in target_set:
        if (x >= target_point).all().item():
            return False
    return True

# Target set should equal to [Tensor(m*1), ..., Tensor(m*1)]
# return: True = dominate some point in target set; False = nondominate all points in target set
def dominate_relation(x:torch.Tensor, target_set:list):
    for sub_vector in target_set:
        if (x >= sub_vector).all().item():
            return False
        elif (x <= sub_vector).all().item() and (x < sub_vector).any().item():
            return True
    return False

def collect_efficient_solutions(candidate_set):
    start = timeit.default_timer()

    # Create Solution objects holding the problem solution and objective values
    Solution = namedtuple("Solution", ["solution", "obj_value"])
    solutions = [
        Solution(solution=object, obj_value=solution.cpu().detach().numpy().reshape(-1))
        for solution in candidate_set
    ]

    # Create an array of shape (solutions, objectives) and compute the non-dominated set
    objective_values_array = np.vstack([s.obj_value for s in solutions])
    mask = paretoset(objective_values_array, sense=["min", "min"])

    # Filter the list of solutions, keeping only the non-dominated solutions
    efficient_solutions = [solution for (solution, m) in zip(solutions, mask) if m]

    stop = timeit.default_timer()

    print("Time: ", round(stop - start, 4), " sec.")

    efficient_set = [
        torch.from_numpy(efficient_solution.obj_value).reshape(-1,1).to(device)
        for efficient_solution in efficient_solutions
    ]
    return efficient_set

# # Candidate set should equal to [Tensor(m*1), ..., Tensor(m*1)]
# def collect_pareto_set(candidate_set:list):

#     # temp_candidate_set = candidate_set.copy()
#     temp_candidate_set = torch.cat(candidate_set, dim=1)
#     temp_candidate_set = torch.unique(temp_candidate_set, dim=1)
#     temp_candidate_set = torch.split(temp_candidate_set, split_size_or_sections=1, dim=1)
        
#     judge_result = []
#     for index, candidate_point in enumerate(temp_candidate_set):
#         if non_dominated_relation(candidate_point, temp_candidate_set[:index]+temp_candidate_set[index+1:]):
#             judge_result.append(candidate_point)
#     return judge_result


# Pareto set(n-points) should equal to [Tensor(m*1), ..., Tensor(m*1)]
# return max(Pareto set ,dim=1)
def get_ref_point(pareto_set: list):
    # [Tensor(m*1), ..., Tensor(m*1)] -> Tensor(m*n) -> Tensor(m,), Tensor(m,)
    values, indices = torch.max(torch.cat(pareto_set, dim=1), dim=1)
    # Tensor(m,) -> Tensor(m,1)
    values = values.reshape(-1, 1)
    return values

# return hypervolume
def calculate_volume(
    lower_corner_value: torch.Tensor, upper_corner_value: torch.Tensor
):
    each_dim_distance = (upper_corner_value - lower_corner_value).reshape(-1)
    cum_prod_result = torch.cumprod(each_dim_distance, dim=0)[-1].item()
    return cum_prod_result


# def plot_pareto_set_2D(pareto_set:list,color:str="r"):
#     fig, ax = plt.subplots(figsize=(6, 4))

#     ax.scatter(
#         torch.cat(pareto_set,dim=1).numpy()[0,:],
#         torch.cat(pareto_set,dim=1).numpy()[1,:],
#         marker=".",
#         s=100,
#         color=color)

def RAAE(true_value, distribution_information:list):
    true_value = true_value.reshape(-1,1)
    
    mean_vector, var_vector = distribution_information
    mean_vector = mean_vector.reshape(-1,1)
    var_vector = var_vector.reshape(-1,1)
    # var_vector[var_vector <= 0] = 1e-10
    std_vector = np.sqrt(var_vector)
    
    n_test = true_value.shape[0]
    diff_vector = abs(true_value - mean_vector)
    diff_vector[diff_vector <= 1e-2] = 0
    
    std_vector[std_vector == 0] = 1e-5
    RAAE = (1 / n_test) * (diff_vector[diff_vector != 0] / std_vector[diff_vector != 0]).sum().item()
    return RAAE

def print_model_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total num of parms: ", pytorch_total_params)
    print("\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "labda" in name:
                print(name, "---->", torch.exp(param.data.cpu()))
            elif "sigma" in name:
                print(name, "---->", param.data.cpu() ** 2)
            else:
                print(name, "---->", param.data.cpu())

if __name__ == '__main__':
    pass