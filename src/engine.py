import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()  # put model in train mode
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for key, val in data.items():
            data[key] = val.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def train_fn(data_loader, model, device):
    model.eval()  # put model in train mode
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for key, val in data.items():
            data[key] = val.to(device)
        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
