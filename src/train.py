import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tqdm import tqdm, trange

from data import batched_iterator, make_dataset
from gpt import GPT

def loss_fn(logits: Tensor, labels):
    log_probs = logits.log_softmax(axis=-1).numpy()
    
    correct = log_probs.gather(idx=labels, dim=-1)[:, 0]
    return -correct.mean()

def train(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    optim,
    steps=10000,  # Adjust this as per the actual training epochs needed
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    allow_jit=True,
):

    def train_step(x, y):
        out = model(x)[:, -1]
        loss = lossfn(out, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        return loss.realize()

    def test_step(x, y):
        out = model(x)[:, -1]
        optim.zero_grad()
        loss = lossfn(out, y)
        return loss.realize()

    if allow_jit:
        train_step = TinyJit(train_step)

    train_losses = []
    test_losses = []
    with Tensor.train():
        for i in (t := trange(steps)):
            train_loss = train_step(X_train, Y_train)
            test_loss = test_step(X_test, Y_test)

            train_losses.append(train_loss.numpy())
            test_losses.append(test_loss.numpy())

            t.set_description(f"train loss: {train_loss.numpy():.2f}, test loss: {test_loss.numpy():.2f}")

    # Plotting the training and testing losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Training and Testing Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, test_losses

if __name__ == "__main__":
    mod = 113
    num_layers = 1
    embed_dim = 128
    vocab_size = mod
    context_length = 3
    num_heads = 4
    batch_size = 8
    num_epochs = 3000
    learning_rate = 1e-3
    wd = 1.0
    train_test_ratio = 0.3

    x_train, y_train, x_test, y_test = make_dataset(train_test_ratio, mod)

    model = GPT(num_layers, embed_dim, vocab_size, context_length, num_heads)

    optimizer = AdamW(get_parameters(model), lr=learning_rate, b1=0.9, b2=0.98, wd=wd)

    print(f"GPT model has {len(get_parameters(model))} parameters")
    train(model, x_train, y_train, x_test, y_test, optimizer, steps=num_epochs, lossfn=loss_fn)
