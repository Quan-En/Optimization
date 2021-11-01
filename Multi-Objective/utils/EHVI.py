
"""
Expected HyperVolume Improvement: for any p-dimension
version: 2021-10-09


- class: EHVI
    -- calculate_iEI(self, distribution_info: list, values_list: list): -> torch.Tensor
    -- calculate_rEI(self, distribution_info: list, values_list: list): -> torch.Tensor
    -- get_ref_points(self, corners_list: list): -> list
    -- calculate_volume(self, corners_value_list: list): -> torch.Tensor
    -- decide_active_region_f(self, pareto_set: list): -> list
    -- get_all_volume_L(self, corners_list: list): -> list

Process:
    Declare `EHVI` class:
        (1) Setting: pareto set, reference point(max boundary).
        (2) Treat pareto set as input to `decide_active_region_f` get all subsection lower corner value and upper corner value.
        (3) Use corners list and `get_ref_points` to get reference point of each subsection.
        (4) Use corners list and `get_all_volume_L` to get volume-L if each subsection.
    Calculation:
        (1) Use `calculate_iEI` to calculate EI if objective functions are independent.
        (2) Use `calculate_rEI` to calculate EI if objective functions are related.
        
- function: argmax_iEHVI(EHVI_operator, y_means_list: list, y_vars_list: list): -> torch.Tensor (1, ), torch.Tensor (n, )
- function: argmax_rEHVI(EHVI_operator, y_means_list: list, y_covars_list: list): -> torch.Tensor (1, ), torch.Tensor (n, )
"""

from itertools import product
import numpy as np
import scipy.stats as ss

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import svd
from Monte_Carlo_Integration.PyTorch.monte_carlo_from_torchquad import MonteCarlo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

from utils.utils import dominate_relation


class EHVI(object):
    def __init__(self, pareto_set:list, boundary_point: torch.Tensor):
        
        self.pareto_set = pareto_set
        self.boundary_point = boundary_point
        
        self.corners_list = self.decide_active_region_f(self.pareto_set)
        self.num_of_area = len(self.corners_list)
        
        self.ref_points_list = self.get_ref_points(self.corners_list)
        self.all_volume_L_list = self.get_all_volume_L(self.corners_list)
    
    def calculate_rEI(self, distribution_info: list):
        #### Transform relative to independent ####
        
        # mean vector(m, 1), covariance matrix(m, m) = distribution information
        mean_vector, covar_matrix = distribution_info
        m = mean_vector.shape[0]

        # avoid standard deviation equal to zero
        zero_condition = covar_matrix.diag() <= 1e-3
        all_zero_condition = zero_condition.all()

        if all_zero_condition.item():
            return torch.Tensor([0]).squeeze().to(device)

        if zero_condition.any().item():
            covar_matrix[zero_condition.diag()] = 1e-3
            # covar_matrix = torch.diag(torch.diag(covar_matrix)) ## only keep diagonal element
        
        u, s, vh = svd(covar_matrix)
        try:
            sigma_neg_square_root = torch.inverse(u @ torch.diag(s).sqrt())
            mean_vector = sigma_neg_square_root @ mean_vector
            std_vector = torch.ones(m, 1).to(device)
        except:
            std_vector = covar_matrix.diag().sqrt().reshape(-1, 1)
        
        # Multi independent Gaussian:  mean(loc), standard deviation(scale)
        multi_indep_normal = Normal(
            loc=torch.zeros(mean_vector.shape, device=device),
            scale=torch.ones(std_vector.shape, device=device)
        )

        # Calculate expected improvement of each area
        delta_indep_list = []

        for i in range(self.num_of_area):

            # Get the ref_point (m, 1)
            ref_point = self.ref_points_list[i]

            # Get the lower corner value(m, 1) & upper corner value(m, 1)
            lower_corner_value, upper_corner_value = self.corners_list[i]
            try:
                lower_corner_value = sigma_neg_square_root @ lower_corner_value
                upper_corner_value = sigma_neg_square_root @ upper_corner_value
            except:
                pass

            # Get volume-L(1, )
            volume_L_value = self.all_volume_L_list[i]

            # Calculate EI of each subsection

            # Standardize corner value
            standardize_lower_corner = (lower_corner_value - mean_vector) / std_vector
            standardize_upper_corner = (upper_corner_value - mean_vector) / std_vector

            # 'Cumulative density' & 'Probability density' at lower corner
            lower_corner_cdf = multi_indep_normal.cdf(standardize_lower_corner)
            lower_corner_log_pdf = multi_indep_normal.log_prob(standardize_lower_corner)
            lower_corner_pdf = lower_corner_log_pdf.exp()

            # 'Cumulative density' & 'Probability density' at upper corner
            upper_corner_cdf = multi_indep_normal.cdf(standardize_upper_corner)
            upper_corner_log_pdf = multi_indep_normal.log_prob(standardize_upper_corner)
            upper_corner_pdf = upper_corner_log_pdf.exp()

            # Cumulative density between [lower corner, upper corner]
            interval_probability_each_dim = upper_corner_cdf - lower_corner_cdf
            interval_probability_each_dim = interval_probability_each_dim.reshape(-1)

            # Product of each dimension cumulative density
            interval_probability = torch.prod(interval_probability_each_dim)

            # Calculate EI
            Q3 = volume_L_value.item() * interval_probability

            Q2 = torch.prod(ref_point - upper_corner_value) * interval_probability

            Q1_left_sides = (
                ref_point - mean_vector
            ) * upper_corner_cdf + std_vector * upper_corner_pdf
            Q1_right_sides = (
                ref_point - mean_vector
            ) * lower_corner_cdf + std_vector * lower_corner_pdf
            Q1 = torch.prod(Q1_left_sides - Q1_right_sides)

            delta_indep = Q1 - Q2 + Q3
            
            delta_indep_list.append(delta_indep)

        return torch.Tensor(delta_indep_list).sum()
        
    def calculate_iEI(self, distribution_info: list):
        #### Assume objective are independent: easy way to solve ####

        # mean vector(m, 1), variance vector(m, 1) = distribution information
        mean_vector, var_vector = distribution_info
        std_vector = var_vector.sqrt()

        # Avoid standard deviation equal to zero
        zero_condition = (std_vector <= 1e-3)
        all_zero_condition = zero_condition.all()
        std_vector[zero_condition] = 1e-3
        
        if all_zero_condition.item():
            return torch.Tensor([0]).squeeze().to(device)
        
        
        # Multi independent Gaussian:  mean(loc), standard deviation(scale)
        multi_indep_normal = Normal(
            loc=torch.zeros(mean_vector.shape, device=device),
            scale=torch.ones(var_vector.shape, device=device)
        )
        # multi_indep_normal = Normal(loc=mean_vector, scale=std_vector)

        # Calculate expected improvement of each area
        delta_indep_list = []

        for i in range(self.num_of_area):

            # Get the ref_point (m, 1)
            ref_point = self.ref_points_list[i]

            # Get the lower corner value(m, 1) & upper corner value(m, 1)
            lower_corner_value, upper_corner_value = self.corners_list[i]

            # Get volume-L(1, )
            volume_L_value = self.all_volume_L_list[i]

            # Calculate EI of each subsection

            # Standardize corner value
            standardize_lower_corner = (lower_corner_value - mean_vector) / std_vector
            standardize_upper_corner = (upper_corner_value - mean_vector) / std_vector

            # 'Cumulative density' & 'Probability density' at lower corner
            lower_corner_cdf = multi_indep_normal.cdf(standardize_lower_corner)
            lower_corner_log_pdf = multi_indep_normal.log_prob(standardize_lower_corner)
            lower_corner_pdf = lower_corner_log_pdf.exp()

            # 'Cumulative density' & 'Probability density' at upper corner
            upper_corner_cdf = multi_indep_normal.cdf(standardize_upper_corner)
            upper_corner_log_pdf = multi_indep_normal.log_prob(standardize_upper_corner)
            upper_corner_pdf = upper_corner_log_pdf.exp()

            # Cumulative density between [lower corner, upper corner]
            interval_probability_each_dim = upper_corner_cdf - lower_corner_cdf
            interval_probability_each_dim = interval_probability_each_dim.reshape(-1)

            # Product of each dimension cumulative density
            interval_probability = torch.prod(interval_probability_each_dim)

            # Calculate EI
            Q3 = volume_L_value.item() * interval_probability

            Q2 = torch.prod(ref_point - upper_corner_value) * interval_probability

            Q1_left_sides = (
                ref_point - mean_vector
            ) * upper_corner_cdf + std_vector * upper_corner_pdf
            Q1_right_sides = (
                ref_point - mean_vector
            ) * lower_corner_cdf + std_vector * lower_corner_pdf
            Q1 = torch.prod(Q1_left_sides - Q1_right_sides)

            delta_indep = Q1 - Q2 + Q3
            
            if (delta_indep<0).item():
                print("delta_indep less than 0: ",delta_indep,
                      "ref_point:", ref_point,
                      "upper_corner_value:",upper_corner_value,
                      "lower_corner_value:",lower_corner_value,
                     "pareto_set:",self.pareto_set)
            delta_indep[delta_indep < 0] = 0

            delta_indep_list.append(delta_indep)

        return torch.Tensor(delta_indep_list).sum()

    def old_calculate_rEI(self, distribution_info: list):
        #### Assume objective are related: numerical integration ####

        # Declare an integrator, here we use the simple, stochastic Monte Carlo integration method
        mc = MonteCarlo()

        # mean vector(m, 1), covariance matrix(m, m) = distribution information
        mean_vector, covar_matrix = distribution_info

        # avoid standard deviation equal to zero
        zero_condition = covar_matrix.diag() <= 1e-3
        all_zero_condition = zero_condition.all()
        if all_zero_condition.item():
            return torch.Tensor([0]).squeeze().to(device)
        # covar_matrix[zero_condition.diag()] = 1e-6
        if zero_condition.any().item():
            covar_matrix[zero_condition.diag()] = 1e-3
            covar_matrix[0,1] = 0
            covar_matrix[1,0] = 0
        
        # covar_matrix_rank = torch.linalg.matrix_rank(covar_matrix).item()
        covar_matrix_det = covar_matrix.det().item()
        if covar_matrix_det < 1e-10:
            covar_matrix[0,1] = 0
            covar_matrix[1,0] = 0
            # covar_matrix = covar_matrix + torch.Tensor([1e-3, 1e-3]).to(device).diag()
            
        std_vector = covar_matrix.diag().sqrt().reshape(-1, 1)

        # Multi Gaussian:  mean(loc), covariance matrix
        multi_normal = MultivariateNormal(
            loc=mean_vector.reshape(-1), covariance_matrix=covar_matrix
        )

        # Mini value of upper corners (use to avoid -inf bound of lower corner)
        min_value, min_index = torch.cat(
            [upper_corner for lower_corner, upper_corner in self.corners_list], dim=1
        ).min(dim=1)
        min_value = min_value.reshape(-1, 1)

        # Calculate expected improvement of each area
        delta_relate_list = []
        sample_N_list = []
        for i in range(self.num_of_area):

            # Get the ref_point(m, 1)
            ref_point = self.ref_points_list[i]

            # Get the lower corner value(m, 1) & upper corner value(m, 1)
            lower_corner_value, upper_corner_value = self.corners_list[i]

            # Get volume-L(1,)
            volume_L_value = self.all_volume_L_list[i]

            # Decide integration domain
            neg_inf_condition = (lower_corner_value == -float("inf"))
            lower_corner_value[neg_inf_condition] = (mean_vector - 5 * std_vector)[
                neg_inf_condition
            ]
            if (lower_corner_value >= upper_corner_value).all().item():
                delta_relate_list.append(torch.Tensor([0]).squeeze().to(device))
                sample_N_list.append(0)
            else:
                integration_domain = torch.cat(
                    [lower_corner_value, upper_corner_value], dim=1
                )
                integration_domain = integration_domain.split(split_size=1, dim=0)
                integration_domain = [
                    sub_int_domain.reshape(-1).tolist()
                    for sub_int_domain in integration_domain
                ]

                # Calculate function
                def Delta_f(x):
                    x = x.to(device)
                    prob_result = multi_normal.log_prob(x).exp()
                    Q1 = prob_result * (ref_point.reshape(-1) - x).prod(dim=1)
                    Q2 = prob_result * (ref_point - upper_corner_value).prod()
                    Q3 = prob_result * volume_L_value.item()
                    delta = Q1 - Q2 + Q3
                    return delta
                sample_N = int(((upper_corner_value - lower_corner_value + 1e-1) * 25).prod().item()+1)
                sample_N_list.append(sample_N)
                delta_relate = mc.integrate(Delta_f, dim=2, N=sample_N, integration_domain=integration_domain, seed=22)
                # delta_relate[delta_relate < 0] = 0

                delta_relate_list.append(delta_relate)
        print("sample size: ", sum(sample_N_list))
        return torch.Tensor(delta_relate_list).sum()
    
    def get_ref_points(self, corners_list: list):
        ref_points_list = []
        num_of_area = len(corners_list)
        for i in range(num_of_area):
            # For each corner (lower index, upper index)
            lower_corner_value, upper_corner_value = corners_list[i]

            # Decide reference point
            ## Get all (upper index) to decide reference point
            candidate_ref_point_list = [
                Ref_upper_corner_value
                for Ref_lower_corner_value, Ref_upper_corner_value in corners_list
                if (Ref_upper_corner_value >= upper_corner_value).all().item()
            ]
#             candidate_ref_point_list = []
#             for pareto_point in self.pareto_set:
#                 if (lower_corner_value == pareto_point).any().item():
#                     candidate_ref_point_list.append(pareto_point)
#             if len(candidate_ref_point_list) <= 1:
#                 candidate_ref_point_list.append(self.boundary_point)

            # [Tensor(m*1), ..., Tensor(m*1)] -> Tensor(m*n) -> Tensor(m,), Tensor(m,)
            ref_point_values, ref_point_indices = torch.max(
                torch.cat(candidate_ref_point_list, dim=1), dim=1
            )
            # Tensor(m,) -> Tensor(m,1)
            ref_point_values = ref_point_values.reshape(-1, 1)

            ref_points_list.append(ref_point_values)
        return ref_points_list

    # return hypervolume
    def calculate_volume(self, corners_value_list: list):
        lower_corner_value, upper_corner_value = corners_value_list

        each_dim_distance = upper_corner_value.reshape(-1) - lower_corner_value.reshape(
            -1
        )
        cum_prod_result = torch.prod(each_dim_distance).item()
        return cum_prod_result

    # Collect each cell corner which is active
    def decide_active_region_f(self, pareto_set: list):

        # n = |pareto_set|
        n = len(pareto_set)
        # m = dimensions of objective function
        m = pareto_set[0].shape[0]

        # neg_inf_tensor = torch.Tensor([float("-inf")]*m).reshape(m,1).to(device)
        neg_inf_tensor = torch.Tensor([-float("inf")] * m).reshape(m, 1).to(device)
        # Get all (upper corner value, lower corner value)

        ## Get unique values list along each dimension (ascending, smallest is negative infinity)
        each_dim_value_list = torch.split(
            torch.cat(pareto_set + [neg_inf_tensor] + [self.boundary_point], dim=1),
            split_size_or_sections=1,
            dim=0,
        )
        each_dim_unique_value_list = [
            torch.unique(value_tensor).tolist() for value_tensor in each_dim_value_list
        ]

        lower_corner_list = list(
            product(
                *[
                    dim_unique_value[:-1]
                    for dim_unique_value in each_dim_unique_value_list
                ]
            )
        )
        upper_corner_list = list(
            product(
                *[
                    dim_unique_value[1:]
                    for dim_unique_value in each_dim_unique_value_list
                ]
            )
        )

        ## Reshape each corner values to torch.Tensor(m*1) and put it on device
        lower_corner_list = [
            torch.Tensor(corner_value).reshape(m, 1).to(device)
            for corner_value in lower_corner_list
        ]
        upper_corner_list = [
            torch.Tensor(corner_value).reshape(m, 1).to(device)
            for corner_value in upper_corner_list
        ]

        # Select active region
        # Problem: pick the region that the lower corner are not dominated or equal to points in pareto_set
        # <=> pick the region that the lower corner are dominate to pareto_set

        corners_list = []
        for lower_corner_value, upper_corner_value in zip(
            lower_corner_list, upper_corner_list
        ):
            if dominate_relation(lower_corner_value, pareto_set):
                corners_list.append((lower_corner_value, upper_corner_value))

        return corners_list

    def get_all_volume_L(self, corners_list: list):

        volume_L_value_list = []

        num_of_area = len(corners_list)
        for i in range(num_of_area):
            # For each corner (lower index, upper index)
            lower_corner_value, upper_corner_value = corners_list[i]

            # Calculate volume-L
            ## Get all (lower index, upper index) for volume-L
            volume_L_corners_list = [
                (L_lower_corner_value, L_upper_corner_value)
                for L_lower_corner_value, L_upper_corner_value in corners_list
                if (upper_corner_value <= L_lower_corner_value).all().item()
                and (upper_corner_value <= L_upper_corner_value).all().item()
            ]

            ## Set memory and start from zero
            volume_L = torch.Tensor([0]).to(device)

            if len(volume_L_corners_list) != 0:
                # Add each sub-volume of L
                for volume_L_corner_values in volume_L_corners_list:
                    volume_L += self.calculate_volume(volume_L_corner_values)

            # Store them to list
            volume_L_value_list.append(volume_L)
        return volume_L_value_list

def argmax_rEHVI(EHVI_operator, y_means: torch.Tensor, y_covars: torch.Tensor):
    
    # y_means = tensor(sample size, 2, 1)
    # y_covars = tensor(sample size, 2, 2)
    num_of_calculate = y_means.shape[0]
    
    EI_result = [
        EHVI_operator.calculate_rEI(
            [
                y_means[i,:,:],
                y_covars[i,:,:]
            ]
        )
        for i in range(num_of_calculate)
    ]
    
    EI_result = torch.Tensor(EI_result)
    argmax_index = torch.argmax(EI_result)
    
    return argmax_index, EI_result

def argmax_iEHVI(EHVI_operator, y_means: torch.Tensor, y_covars: torch.Tensor):
    
    # y_means = tensor(sample size, 2, 1)
    # y_covars = tensor(sample size, 2, 2)
    num_of_calculate = y_means.shape[0]
    
    EI_result = [
        EHVI_operator.calculate_iEI(
            [
                y_means[i,:,:],
                y_covars[i,:,:].diag().reshape(-1,1)
            ]
        )
        for i in range(num_of_calculate)
    ]
    
    EI_result = torch.Tensor(EI_result)
    argmax_index = torch.argmax(EI_result)
    
    return argmax_index, EI_result
    
if __name__ == '__main__':
    pass