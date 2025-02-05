from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Any

def parallel_map(func: Callable, iterable: List[Any], *args, **kwargs) -> List[Any]:
    """
    Map a function to an iterable in parallel
    :param func: Function to map
    :param iterable: Iterable to map the function to
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function
    :return: List of results
    """    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: func(x, *args, **kwargs), iterable))
    return results

def parallel_map_2d(func: Callable, iterable: List[List[Any]], *args, **kwargs) -> List[List[Any]]:
    """
    Map a function to a 2D iterable in parallel
    :param func: Function to map
    :param iterable: 2D iterable to map the function to
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function
    :return: List of results
    """   
    flattened_iterable = [item for sublist in iterable for item in sublist]
    results = parallel_map(func, flattened_iterable, *args, **kwargs)
    return [results[i:i+len(iterable[0])] for i in range(0, len(results), len(iterable[0]))]