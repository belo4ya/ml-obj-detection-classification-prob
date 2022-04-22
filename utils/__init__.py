from __future__ import annotations

from itertools import chain
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from utils.image import *
from utils.label import *
from utils.plot import *
from utils.transforms import *


def write_yaml(path: Path, data: dict):
    with open(path, 'w') as f:
        yaml.dump(data, stream=f)


def train_valid_test_split(*arrays, train_size, valid_size, **kwargs):
    split = train_test_split(*arrays, train_size=train_size, **kwargs)
    train = [split[i] for i in range(0, len(split), 2)]
    rest = [split[i] for i in range(1, len(split), 2)]

    valid_size = valid_size / (1 - train_size)
    split = train_test_split(*rest, train_size=valid_size)
    return list(chain.from_iterable(zip(
        train,
        [split[i] for i in range(0, len(split), 2)],
        [split[i] for i in range(1, len(split), 2)]
    )))
