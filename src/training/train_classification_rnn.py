import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


# TODO: 
#   Implement weight decay?

def train_classification_rnn(model, train_loader, val_loader=None, epochs=20, lr=1e-3, device="cuda"):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)      # (B, num_classes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch} - train loss: {train_loss / len(train_loader):.4f}")

        # optional validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch} - val loss: {val_loss:.4f}")


def evaluate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)
