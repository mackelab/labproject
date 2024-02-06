import numpy as np
import torch
from pyddm.sample import Sample


def filter_roitman_data(
    df,
    coherence,
    animal=0,
    n_trial=400,
    attach_model_mask=False,
    partition=None,
    data_mode="tensor",
):
    """extract specific part of roitman data and put it into a tensor

    Args:
        df (dataframe): roitman data
        coherence (int): in {0,32,64,128, 256,512}
        animal (int, optional): _description_. Defaults to 0.
        n_trial (int, optional): number of random trials to choose.
            Defaults to 400. if 'all': all possible trials are given back.
        attach_model_mask (bool): wheter to attach a model mask
        partition: partition of ddm model. only needed if attach_model_mask==True.
        data_mode: 'pyddm' or 'tensor'
    Returns:
        tensor or pyddm_Sample:  x_i: (rt, decisions) with potentially attached model mask
            OR pyddm Sample
    """

    # extract values
    rt = df[(df["animal"] == animal) & (df["coherence"] == coherence)]["rt"].values / 1000

    decision = df[(df["animal"] == animal) & (df["coherence"] == coherence)]["decision"].values

    if n_trial == "all":
        mask = np.ones(len(rt), dtype=bool)

    else:
        # sample randomly 400 values
        np.random.seed(0)
        mask = np.random.choice(np.arange(len(rt)), size=n_trial, replace=False)  # len(rt),

    if data_mode == "tensor":
        x_i = torch.tensor(np.array([rt[mask], decision[mask]], dtype=np.float32).T)

        if attach_model_mask:
            if partition == None:
                raise (Warning("partition needed."))
            x_i = torch.cat([x_i, torch.ones(len(partition), 2)])
            x_i[-3] = 0
            x_i[-5] = 0
        return x_i

    elif data_mode == "pyddm":
        rt_corr = rt[mask][decision[mask] == 1]
        rt_err = rt[mask][decision[mask] == 0]
        sample = Sample(rt_corr, rt_err)
        return sample
