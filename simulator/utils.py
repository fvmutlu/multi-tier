# External package imports
import numpy as np

# Builtin imports
import json
from collections import deque, defaultdict
from itertools import product
from datetime import timedelta

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def timeDiffPrinter(time_diff: timedelta):
    """
    Convert a timedelta object into a formatted string representation of time difference.

    Args:
        time_diff (timedelta): The time difference to be converted.

    Returns:
        str: A formatted string representing the time difference in hours, minutes, seconds, and microseconds.

    Example:
        >>> time_diff = timedelta(hours=2, minutes=30, seconds=15, microseconds=500000)
        >>> timeDiffPrinter(time_diff)
        '2h 30m 15.500000s'
    """
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds // 60) % 60
    seconds = time_diff.seconds % 60
    microseconds = time_diff.microseconds
    time_components = []
    if hours > 0:
        time_components.append("{}h".format(hours))
    if minutes > 0:
        time_components.append("{}m".format(minutes))
    time_components.append("{}.{}s".format(seconds, microseconds))
    formatted_time_diff = " ".join(time_components)
    return formatted_time_diff


def namedProduct(**items: dict):
    """
    Generates Cartesian product of named items from multiple arguments.

    Args:
        **items: Keyword arguments representing named items with iterable values.

    Yields:
        A generator object that contains all combinations of named items from the input.

    Example 1:
        >>> input_dict = {'a': [1,2], 'b': ['x','y']}
        >>> for combination in namedProduct(**input_dict):
        >>>     print(combination)
        {'a': 1, 'b': 'x'}
        {'a': 1, 'b': 'y'}
        {'a': 2, 'b': 'x'}
        {'a': 2, 'b': 'y'}
    Example 2:
        >>> for combination in namedProduct(a=[1,2], b=['x']):
        >>>     print(combination)
        {'a': 1, 'b': 'x'}
        {'a': 2, 'b': 'x'}
    """
    names = items.keys()
    vals = items.values()
    for res in product(*vals):
        yield dict(zip(names, res))


def namedZip(**items: dict):
    """
    Generates dot product of named items from multiple arguments.

    Args:
        **items: Keyword arguments representing named items with iterable values.

    Yields:
        A generator object that contains all combinations of named items from the input.

    Example 1:
        >>> input_dict = {'a': [1,2], 'b': ['x','y']}
        >>> for combination in namedProduct(**input_dict):
        >>>     print(combination)
        {'a': 1, 'b': 'x'}
        {'a': 2, 'b': 'y'}
    """
    names = items.keys()
    vals = items.values()
    for res in zip(*vals):
        yield dict(zip(names, res))


def invertDict(input_dict: dict):
    """
    Invert a dictionary mapping from keys to sequences of values.

    Args:
        input_dict: The input dictionary to be inverted.

    Returns:
        dict: A new dictionary where the values of the input dictionary become keys,
              and the keys become values, forming lists of keys corresponding to each value.

    Example:
        >>> invertDict({'a': [1, 2], 'b': [2, 3]})
        {1: ['a'], 2: ['a', 'b'], 3: ['b']}
    """
    new_dict = defaultdict(list)
    for key, seq in input_dict.items():
        for item in seq:
            new_dict[item].append(key)
    return dict(new_dict)


def resetDict(input_dict: dict, default_value):
    """
    Resets the values in a dictionary to a given default value.

    Args:
        input_dict: The input dictionary to be reset.
        default_value: Default value to reset every value in the dictionary to.

    Returns:
        dict: A new dictionary with the same keys as those of input_dict
              but all values equal to default_value.

    Example:
    >>> input_dict = {'a': 5, 'b': 12}
    {'a': 5, 'b': 12}
    >>> resetDict(input_dict, 0)
    {'a': 0, 'b': 0}
    """
    return input_dict.fromkeys(input_dict, default_value)


def convertListFieldsToTuples(dictionary):
    """
    Converts list fields in a dictionary to tuples.

    Args:
        dictionary (dict): The dictionary to be processed.

    Returns:
        dict: The dictionary with list fields converted to tuples.
    """
    for key, value in dictionary.items():
        if isinstance(value, list):
            dictionary[key] = tuple(value)
    return dictionary


def randargmax(arr: np.ndarray, axis: int, seed: int = 1):
    """
    Return the indices of the maximum values along an axis of a numpy array, randomly choosing among ties.

    Args:
        arr (numpy.ndarray): The input array.
        axis (int): The axis along which to find the maximum values.
        seed (int): Seed value for random number generation. Default is 1.

    Returns:
        numpy.ndarray: Array of indices of maximum values along the specified axis.

    Example:
        >>> arr = np.array([[1, 2, 3], [4, 2, 1]])
        >>> randargmax(arr, axis=0)
        array([1, 0, 0])
        >>> randargmax(arr, axis=1)
        array([2, 0])
    """
    rng = np.random.default_rng(seed)
    return np.apply_along_axis(
        lambda x: rng.choice(np.where(x == x.max())[0]), axis=axis, arr=arr
    )


class wique:
    """
    A class for maintaining a moving window with a quick update for mean calculation.

    Attributes:
        maxlen (int): The maximum length of the window.
        curlen (int): The current length of the window.
        q (collections.deque): Deque representing the window.
        mean (float): The mean value of the elements in the window.
        sum (float): The sum of the elements in the window.

    Methods:
        __init__(self, maxlen=10): Initializes the wique object with a specified maximum length.
        append(self, x): Adds a new element to the window, updating mean and sum accordingly.
    """

    def __init__(self, maxlen: int = 10):
        """
        Initializes a wique object with a specified maximum length.

        Args:
            maxlen (int): The maximum length of the window. Default is 10.
        """
        self.maxlen = maxlen
        self.curlen = 0
        self.q = deque(maxlen=maxlen)
        self.mean = 0
        self.sum = 0

    def append(self, x: int) -> None:
        """
        Adds a new element to the window, updating mean and sum accordingly.

        Args:
            x: The new element to add to the window.
        """
        if self.curlen < self.maxlen:
            self.sum += x
            self.curlen += 1
            self.mean = self.sum / self.curlen
        else:
            self.sum = self.sum - self.q[0] + x
            self.mean = self.sum / self.maxlen
        self.q.append(x)
