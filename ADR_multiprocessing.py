import multiprocessing as mp
import os

from typing import List, Union
from collections.abc import Iterable

def unwrap_params(list_of_params_dicts : Union[ List[dict], dict ]) -> List[dict]:
    '''
        We recursively go through the parameter dictionary, and
        if any value is an interable, we create a new dictionary for every
        combination of parameters in lists.
    '''
    # If we have a single dictionary, we wrap it in a list
    if isinstance(list_of_params_dicts, dict):
        return unwrap_params([list_of_params_dicts])

    #   We go through the list of dictionaries
    for param_dicts in list_of_params_dicts:

        for key, value in param_dicts.items():
            #  If we find a value that is an iterable, we unpack it
            if isinstance(value, Iterable):
                # We remove the dictionary from the list
                list_of_params_dicts.remove(param_dicts)
                # We create a new dictionary for every value in the iterable
                for v in value:
                    new_dict = param_dicts.copy()
                    new_dict[key] = v
                    # We add the new dictionary to the list
                    list_of_params_dicts.append(new_dict)
                # We recursively call the function to unpack the new dictionaries
                return unwrap_params(list_of_params_dicts)
    # If we don't find any iterable, we
    return list_of_params_dicts

        
def ADR_parameter_span(param_dicts = None):

    from ADR import ADR, ADR_params_dict

    # If no parameter dictionary is provided, we use the default one
    if param_dicts is None:
        param_dicts = ADR_params_dict

    # We unpack the parameter dictionary
    list_of_params_dicts = unwrap_params([param_dicts])

    N_CPU = os.cpu_count()
    # Number of simulations
    n_sims = len(list_of_params_dicts)
    
    # Run simulations in parallel
    with mp.Pool(processes = N_CPU) as pool:
        res = pool.starmap(ADR, list_of_params_dicts, chunksize = n_sims // N_CPU)

    return res

if __name__ == '__main__':

    print(
        unwrap_params(
        {'a' : [1, 2], 'b' : [3, 4], 'c' : 1, 'd' : [2, 3, 4]}
        )
    )