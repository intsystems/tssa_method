""" module contains algorithms of discrete optimization to find grouping with less hankelization residual (many signals case)
"""
import numpy as np
import scipy.linalg as linalg

from copy import deepcopy
from typing import Union, List

from ..ssa_classic import SSA_classic
from ..m_ssa import m_SSA
from ..t_ssa import t_SSA
from .ssa_classic_partition import local_search_partitioning as loc_search_classic, reset_best_results, \
                                    get_best_found_grouping, get_best_integral_hankel_resid, \
                                    build_random_initial_grouping, \
                                    dichotomy_partition as dichotomy_partition_classic


def local_search_partitioning(ssa_obj: Union[m_SSA, t_SSA], init_grouping_list: list, steps_limit: int=5) -> tuple:
    """local search algorithm of finding good grouping. Proceeds until the steps limit or inability to find better solution in
        vicinity

    :param Union[m_SSA, t_SSA] ssa_obj: ssa object for multidimensional signals
    :param list init_grouping_list: list of init grouping for every signal in ssa object; if some element is None,
                                                                                         then init grouping will be random
    :param int steps_limit: number of steps to proceed, defaults to 5
    :return tuple: best grouping list and mean hankel residual list for every signal
    """
    best_grouping_list = []
    best_hankel_resids_list = []

    # technical variable for mSSA
    temp = 0
    # iterating over all signals
    for i in range(len(ssa_obj.t_s_list)):
        # technical variable for mSSA
        cur_sig_len = len(ssa_obj.t_s_list[i])

        # create ssa_classic object
        ssa_clas_temp = SSA_classic(ssa_obj.t_s_list[i], ssa_obj.L)
        
        if isinstance(ssa_obj, t_SSA):
            ssa_clas_temp.weights, ssa_clas_temp._left_factors, ssa_clas_temp._right_factors = ssa_obj.get_available_factors(i)
        if isinstance(ssa_obj, m_SSA):
            ssa_clas_temp.weights, ssa_clas_temp._left_factors, ssa_clas_temp._right_factors = \
                    ssa_obj.weights, ssa_obj._left_factors, ssa_obj._right_factors[:, temp:temp + (cur_sig_len - ssa_obj.L + 1)]
            
        temp += cur_sig_len - ssa_obj.L + 1

        # debug
        print(f'Local search for signal {i + 1}')

        # use partition algorithm for 1d signal
        loc_search_classic(ssa_clas_temp, init_grouping_list[i], steps_limit)

        # save results
        best_grouping_list.append(get_best_found_grouping())
        best_hankel_resids_list.append(get_best_integral_hankel_resid())

        # reset found result in classic_partitioning variables
        reset_best_results()
 
    return best_grouping_list, best_hankel_resids_list


def NextDichotomyPartition(ssa_obj: Union[m_SSA, t_SSA], init_grouping: list):
    """generator for consequtive dichotomy partitions

    Args:
        ssa_obj (Union[m_SSA, t_SSA]): 
        init_grouping (list): grouping to start disecting with
    """
    num_signals = len(ssa_obj.t_s_list)

    # in case we don't have grouping => build first partition on two groups for each signal
    if init_grouping is None:
        signals_index_array, hankel_resid_array = _InitSignalsArrays(ssa_obj)

        # for each signal we keep a list of components to futher decompose
        # user is able to control this components with every iteration of the algorithm
        components_to_decompose = [[0] for _ in range(num_signals)]

        # build first partition on two groups for each signal
        new_signals_index_array, new_hankel_resid_array = _MakePartition(
            ssa_obj, 
            signals_index_array, 
            hankel_resid_array, 
            components_to_decompose
        )

        # obtain group numbers to further disect for each signal
        # return grouping on two
        components_to_decompose = yield new_signals_index_array, new_hankel_resid_array

        # update containers
        signals_index_array = new_signals_index_array
        hankel_resid_array = new_hankel_resid_array
    # if we have grouping to start with => get group numbers to further disect
    else:
        signals_index_array = init_grouping
        hankel_resid_array = []

        # get hankel residuals for this grouping
        for i in range(num_signals):
            ssa_obj.set_factors_grouping(signals_index_array[i], i)
    
        _, hankel_resid_array, _ = ssa_obj.decompose_signals()

        # obtain group numbers to further disect for each signal
        components_to_decompose = yield

    while True:
        # build partition
        new_signals_index_array, new_hankel_resid_array = _MakePartition(
            ssa_obj, 
            signals_index_array, 
            hankel_resid_array, 
            components_to_decompose
        )

        # obtain new group numbers to further disect for each signal
        components_to_decompose = yield new_signals_index_array, new_hankel_resid_array

        # update containers
        signals_index_array = new_signals_index_array
        hankel_resid_array = new_hankel_resid_array


def _InitSignalsArrays(ssa_obj: Union[m_SSA, t_SSA]):
    """
        return signal's groups and hankel residuals for unpartitioned signals
    """
    num_signals = len(ssa_obj.t_s_list)

    # obtain available indices for each signal
    # also initialize hankel_residuals for each signal/group
    if (isinstance(ssa_obj, m_SSA)):
        hankel_resid_array = [[0] for _ in range(num_signals)]

        # signals_index_array has 3 indices: signal_number, group of indices, index within group
        signals_index_array = [[list(range(ssa_obj.weights.size))] for _ in range(len(ssa_obj.t_s_list))]
    else:
        hankel_resid_array = [[0] for _ in range(num_signals)]

        # signals_index_array has 3 indices: signal_number, group of indices, index within group
        signals_index_array = []

        for sig_num in range(num_signals):
            signal_weights, _, _ = ssa_obj.get_available_factors(sig_num)
            
            signals_index_array.append([list(range(signal_weights.size))])

    return signals_index_array, hankel_resid_array


def _MakePartition(
        ssa_obj: Union[m_SSA, t_SSA], 
        signals_index_array : list, 
        hankel_resid_array: list,
        components_to_decompose: list
):
    """
        Make one iteration of dichotomy partition
    """
    num_signals = len(ssa_obj.t_s_list)

    # containers for new groupings and residuals
    new_hankel_resid_array = [[] for _ in range(num_signals)]
    new_signals_index_array = [[] for _ in range(num_signals)]

    for sig_num in range(num_signals):
        # get complementery 1d-ssa initialized with current signal data
        classic_ssa_obj = _init_classic_ssa(ssa_obj, sig_num)
        
        for indx_group in range(len(signals_index_array[sig_num])):
            cur_group = signals_index_array[sig_num][indx_group]
            cur_resid = hankel_resid_array[sig_num][indx_group]
            
            # do not decompose this group => just copy it
            if (indx_group not in components_to_decompose[sig_num]) or (len(cur_group) < 2):
                new_signals_index_array[sig_num].append(cur_group)
                new_hankel_resid_array[sig_num].append(cur_resid)
            # decompose current group into two new
            else:
                hank_res, grouping = dichotomy_partition_classic(classic_ssa_obj, cur_group)

                new_signals_index_array[sig_num].append(grouping[0])
                new_signals_index_array[sig_num].append(grouping[1])

                new_hankel_resid_array[sig_num].append(hank_res[0])
                new_hankel_resid_array[sig_num].append(hank_res[1])

    return new_signals_index_array, new_hankel_resid_array


"""
def NextDichotomyPartition(ssa_obj: Union[m_SSA, t_SSA], init_grouping: list):
    num_signals = len(ssa_obj.t_s_list)

    # obtain available indices for each signal
    # also initialize hankel_residuals for each signal/group
    if (isinstance(ssa_obj, m_SSA)):
        hankel_resid_array = [[0] for _ in range(num_signals)]

        # signals_index_array has 3 indices: signal_number, group of indices, index within group
        signals_index_array = [[list(range(ssa_obj.weights.size))] for _ in range(len(ssa_obj.t_s_list))]
    else:
        hankel_resid_array = [[0] for _ in range(num_signals)]

        # signals_index_array has 3 indices: signal_number, group of indices, index within group
        signals_index_array = []

        for sig_num in range(num_signals):
            signal_weights, _, _ = ssa_obj._get_available_factors(sig_num)
            
            signals_index_array.append([list(range(signal_weights.size))]) 

    # for each signal we keep a list of components to futher decompose
    # user is able to control this components with every iteration of the algorithm
    components_to_decompose = [[0] for _ in range(num_signals)]

    while True:
        # containers for new groupings and residuals
        new_hankel_resid_array = [[] for _ in range(num_signals)]
        new_signals_index_array = [[] for _ in range(num_signals)]

        for sig_num in range(num_signals):
            # get complementery 1d-ssa initialized with current signal data
            classic_ssa_obj = _init_classic_ssa(ssa_obj, sig_num)
            
            for indx_group in range(len(signals_index_array[sig_num])):
                cur_group = signals_index_array[sig_num][indx_group]
                cur_resid = hankel_resid_array[sig_num][indx_group]
                
                # do not decompose this group => just copy it
                if (indx_group not in components_to_decompose[sig_num]) or (len(cur_group) < 2):
                    new_signals_index_array[sig_num].append(cur_group)
                    new_hankel_resid_array[sig_num].append(cur_resid)
                # decompose current group into two new
                else:
                    hank_res, grouping = dichotomy_partition_classic(classic_ssa_obj, cur_group)

                    new_signals_index_array[sig_num].append(grouping[0])
                    new_signals_index_array[sig_num].append(grouping[1])

                    new_hankel_resid_array[sig_num].append(hank_res[0])
                    new_hankel_resid_array[sig_num].append(hank_res[1])

        # return current groups for signals and mean hankel residuals
        # and get lists of components to futher decompose
        components_to_decompose = yield new_signals_index_array, new_hankel_resid_array

        # update containers
        signals_index_array = new_signals_index_array
        hankel_resid_array = new_hankel_resid_array
"""


def dichotomy_partition(ssa_obj: Union[m_SSA, t_SSA], indices_sets: List[List]):
    """get's subset of indecies (indices_set) and find best partition in 2 groups for this subset
         (in terms of mean hankel error). Do it for every signal for m-d ssa object

    Args:
        ssa_obj (Union[m_SSA, t_SSA]): 
        indices_sets (List[List]): indecies subsets for each signal

    Returns:
        tuple: (best hankel error, best grouping)
    """
    num_signals = len(ssa_obj.t_s_list)

    # containers for found results for each signal
    best_grouping_lists = []
    best_mean_hankels = []

    for i in range(num_signals):
        # get complementery 1d-ssa initialized with current signal data
        classic_ssa_obj = _init_classic_ssa(ssa_obj, i)

        cur_hank_err, cur_grouping = dichotomy_partition_classic(classic_ssa_obj, indices_sets[i])
        best_mean_hankels.append(cur_hank_err)
        best_grouping_lists.append(cur_grouping)

    return (best_mean_hankels, best_grouping_lists)


def _init_classic_ssa(ssa_obj: Union[m_SSA, t_SSA], sig_num: int):
    """return initialized 1-d ssa object from the given m-d ssa object and one of its signals

    Args:
        ssa_obj (Union[m_SSA, t_SSA]): source object
        sig_num (int): with wich signal to init classic ssa object

    Returns:
        SSA_Classic
    """
    classic_ssa_obj = SSA_classic(ssa_obj.t_s_list[sig_num], ssa_obj.L)

    if type(ssa_obj) is m_SSA:
        ssa_obj : m_SSA = ssa_obj
        classic_ssa_obj.weights = ssa_obj.weights
        classic_ssa_obj._left_factors = ssa_obj._left_factors
        classic_ssa_obj._right_factors = ssa_obj._right_factors[:, sig_num * ssa_obj.L : (sig_num + 1) * ssa_obj.L]
    else:
        ssa_obj : t_SSA = ssa_obj

        classic_ssa_obj.weights, \
        classic_ssa_obj._left_factors, \
        classic_ssa_obj._right_factors = ssa_obj.get_available_factors(sig_num)                           
        
    return classic_ssa_obj
                                                                                    


