# src/train_pytorch.py (core)
import torch
from torch import nn, optim

def train_model(model, train_loader, val_loader, device='cpu', epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_acc, val_loss = eval_model(model, val_loader, device, criterion)
        print(f"Epoch {ep}/{epochs} train_loss {train_loss:.4f} train_acc {train_acc:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
    return model

def eval_model(model, loader, device='cpu', criterion=None):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if criterion:
                running_loss += criterion(out, yb).item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    avg_loss = running_loss/total if criterion else 0.0
    return correct/total, avg_loss