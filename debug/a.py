import torch
from torch import nn, optim
def main():
    model = nn.Linear(10,10)
    adam = optim.Adam(model.parameters())
    loss = ( model(torch.ones(10)) - model(torch.ones(10)) ).sum()
    print(f"DEBUGPRINT[1]: a.py:6: loss={loss}")
    adam.zero_grad()
    loss.backward()
    adam.step()
    pass


if __name__ == "__main__":
    main()
