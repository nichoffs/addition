import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tqdm import tqdm, trange


def logit_dataset(
    model,
    X_ds,
    Y_ds,
    allow_jit=True,
):
    def forward(x, y):
        out = model(x)[:, -1]
        logits = out.cast(dtypes.float64).gather(idx=Y_ds, dim=-1)[:, 0]
        return logits.numpy()

    if allow_jit:
        train_step = TinyJit(forward)

    predictions = forward(X_ds, Y_ds)

    return X_ds[:, :2], predictions
