from visualizations import plot_img_and_recon

def get_latent_vectors(model, loader, device):
    model.eval()

    all_vectors = []

    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            mu, _ = model.encoder(img)
            all_vectors.append(mu.cpu())
    return torch.cat(all_vectors, dim=0)


def add_smile_to_img(model, img, smile_vector, device, alfa=1.0):
    img = img.unsqueeze(0).to(device)
    smile_vector = smile_vector.unsqueeze(0).to(device)
    mu, log_var = model.encoder(img)
    mu += (smile_vector * alfa)

    smile_img = model.decoder(mu)
    smile_img = smile_img.detach().cpu().squeeze()

    plot_img_and_recon(img.squeeze().cpu(), smile_img)


