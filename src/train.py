from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tqdm import tqdm, trange

from data import batched_iterator, make_dataset
from gpt import GPT


def train(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    optim,
    steps=100,  # Only one step is needed for full batch training
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    allow_jit=True,
):
    def train_step(x, y):
        # network
        out = model(x)[:, -1]
        print(f"input train shape {x.shape}")
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("finished step!")
        return loss.realize()

    def test_step(x, y):
        print("testing...")
        out = model(x)[:, -1]
        loss = lossfn(out, y)
        print(loss.shape)
        return loss.realize()

    train_losses = []
    test_losses = []
    with Tensor.train():
        for _ in (t := trange(steps)):
            train_loss = train_step(X_train, Y_train)
            train_losses.append(train_loss)
            t.set_description(f"train loss: {train_loss.numpy():.2f}")

    # Move testing outside of the Tensor.train() context
    # for _ in (t := trange(steps)):
    #     print("starting test")
    #     test_loss = test_step(X_test, Y_test)
    #     test_losses.append(test_loss)
    #     t.set_description(f"test loss: {test_loss:.2f}")

    # return [train_losses, test_losses]


if __name__ == "__main__":
    mod = 113
    num_layers = 1
    embed_dim = 128
    vocab_size = mod
    context_length = 3
    num_heads = 4
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    wd = 1.0
    train_test_ratio = 0.3

    x_train, y_train, x_test, y_test = make_dataset(train_test_ratio, mod)

    model = GPT(num_layers, embed_dim, vocab_size, context_length, num_heads)

    optimizer = AdamW(get_parameters(model), lr=learning_rate, b1=0.9, b2=0.98, wd=wd)

    print(f"GPT model has {len(get_parameters(model))} parameters")
    train(model, x_train, y_train, x_test, y_test, optimizer)
