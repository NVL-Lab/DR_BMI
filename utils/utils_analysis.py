from typing import Optional

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def increase_percent(a: float, b: float) -> float:
    """given 2 values obtain the percent of increase"""
    return ((a - b) / b) * 100
