from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from data import batched_iterator, make_dataset
from gpt import GPT

mod = 113
num_layers = 3
embed_dim = 128
vocab_size = mod
context_length = 16
num_heads = 4
batch_size = 8
num_epochs = 10
learning_rate = 1e-3
wd = 1.0
train_test_ratio = 0.3

x_train, y_train, x_test, y_test = make_dataset(train_test_ratio, mod)

model = GPT(num_layers, embed_dim, vocab_size, context_length, num_heads)

optimizer = AdamW(get_parameters(model), lr=learning_rate, b1=0.9, b2=0.98, wd=wd)

for epoch in range(num_epochs):
    total_loss = 0
    with Tensor.train():
        for x_batch, y_batch in batched_iterator(x_train, y_train, batch_size):
            optimizer.zero_grad()
            loss = model(x_batch)[:, -1, :].sum()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")
