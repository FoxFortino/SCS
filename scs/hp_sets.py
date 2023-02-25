from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import numpy as np
import learn


def dev0():
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_feed_forward_layers"] = [1, 2, 3]
    hp["feed_forward_layer_size"] = [128, 1024, 4096]

    return ParameterGrid(hp)


def aug1():
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["train_frac"] = [0.50, 0.60, 0.70, 0.80]
    hp["noise_scale"] = np.geomspace(0.1, 1, num=6)
    hp["spike_scale"] = np.geomspace(0.1, 5, num=6)
    hp["max_spikes"] = [3, 5]

    return ParameterGrid(hp)


def aug2():
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["train_frac"] = np.arange(0.1, 0.9, 0.025)

    return ParameterGrid(hp)
