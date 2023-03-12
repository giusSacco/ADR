import multiprocessing as mp
import os

from typing import List, Union
from collections.abc import Iterable

def unwrap_params(list_of_params_dicts : Union[ List[dict], dict ]) -> List[dict]:

    '''
        We recursively go through the parameter dictionary
        If any value is an iterable, we unpack it
    '''

    if isinstance(list_of_params_dicts, dict):
        return unwrap_params([list_of_params_dicts])

    for param_dicts in list_of_params_dicts:
        for key, value in param_dicts.items():
            if isinstance(value, Iterable):
                list_of_params_dicts.remove(param_dicts)
                for v in value:
                    new_dict = param_dicts.copy()
                    new_dict[key] = v
                    list_of_params_dicts.append(new_dict)
                
                return unwrap_params(list_of_params_dicts)

    return list_of_params_dicts

        
def ADR_parameter_span(param_dicts):

    from ADR import ADR, ADR_params_dict

    list_of_params_dicts = unwrap_params([param_dicts])

    N_CPU = os.cpu_count()
    n_sims = len(list_of_params_dicts)

    with mp.Pool(processes = N_CPU) as pool:

        res = pool.starmap(ADR, list_of_params_dicts, chunksize = n_sims // N_CPU)

    return res

if __name__ == '__main__':

    print(
        unwrap_params(
        {'a' : [1, 2], 'b' : [3, 4], 'c' : 1, 'd' : [2, 3, 4]}
        )
    )