from tinygrad import Tensor, dtypes
from math import prod


# returns x_train,y_train,x_test,y_test
def make_dataset(train_test_ratio, mod):
    ds_len = mod * mod
    # each have shape 12769

    # [ [0,1,2,..,mod,0,1,2,...mod] ] mod times
    a = Tensor.arange(mod, dtype=dtypes.int).repeat((mod, 1)).flatten(0, -1).unsqueeze(0)
    # [ [0,0,0,...,1,1,1,...,112,112,112] ]
    b = Tensor.arange(mod,dtype=dtypes.int).unsqueeze(-1).repeat((1, mod)).flatten(0, -1).unsqueeze(0)
    # [ [113, 113, 113,...,113, 113] ]
    equals = Tensor.full((ds_len), mod).unsqueeze(0)
    sum = a+b
    products = sum.div(mod).floor() * mod
    targets = sum - products

    ds = a.cat(b, equals, dim=0).T

    indices = Tensor.randint(
        ds_len,
        low=0,
        high=ds_len,
    )

    ds_shuffled = ds[indices].cast(dtypes.float)
    targets_shuffled = targets[:,indices].cast(dtypes.float).reshape(prod(targets.shape),1)

    # print(ds_shuffled.numpy(), f"{targets_shuffled.numpy()=}")
    train_cutoff = int(train_test_ratio * ds_len)

    return (
        ds_shuffled[:train_cutoff],
        targets_shuffled[:train_cutoff],
        ds_shuffled[train_cutoff:],
        targets_shuffled[train_cutoff:],
    )


def batched_iterator(x, y, batch_size):
    dataset_size = x.shape[0]
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        yield (x[start:end], y[start:end])

x_train, y_train, x_test, y_test = make_dataset(.3, 113)