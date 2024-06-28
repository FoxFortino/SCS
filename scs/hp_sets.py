from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import numpy as np
import scs_config as scsc

default_hyper_parameters = {
    "phase_range": (-20, 50),
    "ptp_range": (0.1, 100),
    "wvl_range": (4500, 7000),
    "train_frac": 0.65,
    "noise_scale": 0.1,
    "spike_scale": 1.0,
    "max_spikes": 3,
    "random_state": None,
    
    "model_type": "feed_forward",
    "add_dim": False,
    "swap": False,
    "units": [1024, 1024, 1024],
    "activation": "relu",
    "dropout": 0.1,
    
    # "model_type": "transformer_encoder",
    # "add_dim": True,
    # "swap": True,
    # "encoder_blocks": 1,
    # "encoder_heads": 8,
    # "encoder_key_dim": 32,
    # "encoder_proj_dim": 2048,
    # "encoder_dropout_attention": 0.1,
    # "encoder_dropout_projection": 0.1,
    # "feed_forward_units": [1024, 1024, 1024],
    # "feed_forward_activation": "relu",
    # "feed_forward_dropout": 0.1,

    "lr_schedule": "constant_lr",
    "lr0": 1e-5,
    "epochs": 10_000,
    "batch_size": 32,
}


def T0():
    """
    
    """
    hp = deepcopy(default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["model_type"] = "transformer_encoder"
    hp["add_dim"] = True
    hp["swap"] = True
    hp["encoder_blocks"] = 1
    hp["encoder_heads"] = 8
    hp["encoder_key_dim"] = 32
    hp["encoder_proj_dim"] = 512
    hp["encoder_dropout_attention"] = 0.3
    hp["encoder_dropout_projection"] = 0.3
    hp["feed_forward_units"] = [1024, 1024, 1024]
    hp["feed_forward_activation"] = "relu"
    hp["feed_forward_dropout"] = 0.3

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = range(25)
    hp["encoder_blocks"] = np.arange(8)

    return ParameterGrid(hp)



def ff_test1():
    """
    
    """
    hp = deepcopy(default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["dropout"] = 0

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = range(20)
    hp["units"] = [
        [32], 2 * [32], 3 * [32], 4 * [32], 5 * [32], 
        [64], 2 * [64], 3 * [64], 4 * [64], 5 * [64], 
        [128], 2 * [128], 3 * [128], 4 * [128], 5 * [128], 
        # [256], 2 * [b256], 3 * [256], 4 * [256], 5 * [256], 
        # [512], 2 * [512], 3 * [512], 4 * [512], 5 * [512], 
        # [1024], 2 * [1024], 3 * [1024], 4 * [1024], 5 * [1024], 
    ]

    return ParameterGrid(hp)


def ff_test0():
    """
    Dropout doesn't seem to have a big impact here. Set it to 0 for future tests then try again once architecture is set.
    
    """
    hp = deepcopy(default_hyper_parameters)

    # ----- Change default values here ----- #s

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = range(25)
    hp["dropout"] = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    return ParameterGrid(hp)




def newdev1():
    """

    """
    hp = deepcopy(default_hyper_parameters)

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.65

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = range(25)
    hp["dropout"] = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    # hp["units"] = [
    #     [32], 2 * [32], 3 * [32], 4 * [32], 5 * [32], 
    #     [64], 2 * [64], 3 * [64], 4 * [64], 5 * [64], 
    #     [128], 2 * [128], 3 * [128], 4 * [128], 5 * [128], 
    #     [256], 2 * [256], 3 * [256], 4 * [256], 5 * [256], 
    #     [512], 2 * [512], 3 * [512], 4 * [512], 5 * [512], 
    #     [1024], 2 * [1024], 3 * [1024], 4 * [1024], 5 * [1024], 
    # ]

    return ParameterGrid(hp)




def newdev0():
    """

    """
    hp = deepcopy(default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["lr0"] = 7.27895384e-04
    hp["dropout"] = 0

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = range(25)
    hp["train_frac"] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    # hp["units"] = [
    #     [32], 2 * [32], 3 * [32], 4 * [32], 5 * [32], 
    #     [64], 2 * [64], 3 * [64], 4 * [64], 5 * [64], 
    #     [128], 2 * [128], 3 * [128], 4 * [128], 5 * [128], 
    #     [256], 2 * [256], 3 * [256], 4 * [256], 5 * [256], 
    #     [512], 2 * [512], 3 * [512], 4 * [512], 5 * [512], 
    #     [1024], 2 * [1024], 3 * [1024], 4 * [1024], 5 * [1024], 
    # ]

    return ParameterGrid(hp)



def dev9():
    """
    Trying something new here. Testing at R = 100.

    Still feeling like it is not learning anything. Try a projection to a higher dimension at the beginning of the model.
    """
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.7

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 8
    hp["key_dim"] = 8
    hp["filters"] = 512

    hp["initial_projection"] = 100

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(25)
    hp["num_transformer_blocks"] = np.arange(8)

    return ParameterGrid(hp)



def dev8():
    """
    Trying something new here. Testing at R = 100.
    """
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.7

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 8
    hp["key_dim"] = 8
    hp["filters"] = 512

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(25)
    hp["num_transformer_blocks"] = np.arange(8)

    return ParameterGrid(hp)



def dev7():
    """
    Still seems like model does not improve with increasing number of transformer blocks...
    """
    hp = deepcopy(scsc.default_hyper_parameters)
    
    hp["newmodel"] = True
    hp["vaswani"] = True

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.7

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 8
    hp["key_dim"] = 64
    hp["filters"] = 2048

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(40)
    hp["num_transformer_blocks"] = [0, 2, 4, 6, 8]

    return ParameterGrid(hp)



def dev6():
    """
    After replicating the results from the keras example for time-series classification with transformers (which is what I followed to make a transformer in the first place), I found that their results do not change at all when varying the number of transformer blocks.
    
    I will continue to use the hyper-parameters that I foudn to give the best results when not using transformers, but I expect that num_heads and key_dim will be actually importantfor the success of the model so I am choosing default values that I think make sense.
    
    For this test I will vary the number of transformer blocks to see what happens.
    """
    hp = deepcopy(scsc.default_hyper_parameters)
    
    hp["newmodel"] = True
    hp["vaswani"] = True

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    # hp["num_feed_forward_layers"] = 3
    hp["num_feed_forward_layers"] = 1
    # hp["feed_forward_layer_size"] = 1024
    hp["feed_forward_layer_size"] = 128
    # hp["dropout_feed_forward"] = 0.7
    hp["dropout_feed_forward"] = 0.1

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 8
    hp["key_dim"] = 64
    hp["filters"] = 512

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(10)
    hp["num_transformer_blocks"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    return ParameterGrid(hp)



def dev5():
    """
    Result: This test showed that adding more transforming blocks did nothing
    """
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.7

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 64
    hp["key_dim"] = 4

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(10)
    hp["num_transformer_blocks"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    return ParameterGrid(hp)



def dev4():
    """
    Result: Dropout of 0.7 seems to be give the most consistent and best results for these default args.
    """
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #s
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 7.27895384e-04

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.1

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 64
    hp["key_dim"] = 4

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["retry"] = np.arange(10)
    hp["dropout_feed_forward"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    return ParameterGrid(hp)


def dev3():
    """
    Result: lr0 = 7.27895384e-04 seems better.
    """
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 1e-5

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.1

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 64
    hp["key_dim"] = 4

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["lr0"] = np.geomspace(1e-2, 1e-5, num=30)
    hp["retry"] = np.arange(10)

    return ParameterGrid(hp)



def dev2():
    """
    Result: Totally inconclusives. I think the model needs more transformer blocks. Alternatively, this could be telling us that the transformer is not working or that there is a bug.
    """
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

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
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_transformer_blocks"] = 0

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_feed_forward_layers"] = [1, 2, 3]
    hp["feed_forward_layer_size"] = [128, 1024, 4096]

    return ParameterGrid(hp)


def test():
    hp = deepcopy(scsc.default_hyper_parameters)

    # ----- Change default values here ----- #
    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["train_frac"] = 0.80
    hp["noise_scale"] = 0.15848931924611134
    hp["spike_scale"] = 1.045639552591273
    hp["max_spikes"] = 3
    hp["lr0"] = 1e-3

    hp["num_feed_forward_layers"] = 3
    hp["feed_forward_layer_size"] = 1024
    hp["dropout_feed_forward"] = 0.1

    hp["num_transformer_blocks"] = 0
    hp["num_heads"] = 64
    hp["key_dim"] = 4

    # ----- Reformat dictionary ----- #
    hp = {key: [val] for key, val in hp.items()}

    # ----- Add lists of values to try here ----- #
    hp["num_feed_forward_layers"] = [1, 2, 3]
    hp["feed_forward_layer_size"] = [32, 64, 128]

    return ParameterGrid(hp)
