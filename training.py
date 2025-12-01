import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from vae import VAE
import torch.nn.functional as F
from visualizations import plot_img_and_recon

def elbo_loss(x, x_recon, mu, log_var):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5*(torch.sum(1+log_var-mu.pow(2)-log_var.exp()))

    return BCE + KLD

def train_epoch(model: VAE, train_loader: dataloader, optimizer: torch.optim, device):
    model.train()
    total_loss = 0.0
    for _, (img, _) in enumerate(train_loader):

        img = img.to(device)

        x, mu, log_var = model(img)

        loss = elbo_loss(img, x, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(train_loader.dataset)
    return total_loss

def eval_epoch(model: VAE, test_loader: dataloader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for _, (img, _) in enumerate(test_loader):
            img = img.to(device)
            x, mu, log_var = model(img)
            loss = elbo_loss(img, x, mu, log_var)
            total_loss += loss.item()

        total_loss /= len(test_loader.dataset)

        return total_loss

def train_vae(model: VAE, train_loader:dataloader, test_loader: dataloader, optimizer, device, n_epochs):
    model.to(device)
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = eval_epoch(model, test_loader, device)
        print(f"Epoch: {epoch+1}/{n_epochs} train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")

        if epoch % 10 == 0:
            img, _ = next(iter(test_loader))
            img = img.to(device)
            recon, _, _ = model(img)
            plot_img_and_recon(img[0], recon[0])

    return model
