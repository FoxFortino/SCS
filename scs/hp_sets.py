from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import numpy as np
import learn


def dev3():
    """
    Result:
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50
    hp["noise_scale"] = 0.1
    hp["spike_scale"] = 6.1
    hp["max_spikes"] = 5
    hp["lr0"] = 1e-5
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.5

    hp["key_dim"] = 64

    # hp["num_transformer_blocks"] = 1

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_transformer_blocks"] = [1, 2, 3, 4, 5, 6]
    hp["num_heads"] = [4, 8, 16, 32, 64]

    return ParameterGrid(hp)



def dev2():
    """
    Result: Totally inconclusives. I think the model needs more transformer blocks. Alternatively, this could be telling us that the transformer is not working or that there is a bug.
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50
    hp["noise_scale"] = 0.1
    hp["spike_scale"] = 6.1
    hp["max_spikes"] = 5
    hp["lr0"] = 1e-5
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.5

    hp["num_transformer_blocks"] = 1

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_heads"] = [1, 2, 4, 8, 16, 32, 64]
    hp["key_dim"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    return ParameterGrid(hp)


def dev1():
    """
    Result: Seems like 3 layers of 1024 with a dropout of 0.5 is a good place to be for the feed_forward part of the model
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50
    hp["noise_scale"] = 0.1
    hp["spike_scale"] = 6.1
    hp["max_spikes"] = 5
    hp["lr0"] = 1e-5

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_feed_forward_layers"] = [1, 2, 3, 4, 5]
    hp["feed_forward_layer_size"] = [64, 128, 256, 512, 1024, 2048, 4096]
    hp["dropout_feed_forward"] = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    return ParameterGrid(hp)


def aug5():
    """
    Result: 1e-5 is not too large to allow the maxf1 to get much larger than the maxf1_at_min_valloss, but not too small that training takes forever. Generally inconclusive though, there is a lot of noise in the results
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50
    hp["noise_scale"] = 0.1
    hp["spike_scale"] = 6.1
    hp["max_spikes"] = 5

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["lr0"] = np.geomspace(1e-2, 1e-7, num=100)

    return ParameterGrid(hp)



def aug4():
    """
    Result: Very complicated. Trnf1 goes down with high max_spikes and high_spike_scale. tstf1 is noisy and inconclusive. trnloss predictably goes up with high max_spikes and high spike_scale but tstloss shows the opposite behavior with min tstloss occuring at max_spikes=5 and spike_scale=6.1.
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50
    hp["noise_scale"] = 0.1

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["spike_scale"] = np.linspace(0.1, 8, num=26)
    hp["max_spikes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    return ParameterGrid(hp)


def aug3():
    """
    Result: noise_scale of 0.1 seems good
    """
    hp = deepcopy(learn.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.50

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["noise_scale"] = np.geomspace(0.01, 2, num=51)

    return ParameterGrid(hp)


def aug2():
    """
    Result: train_frac of 0.50 is probably good enough. Lower would likely even be fine.
    """
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
