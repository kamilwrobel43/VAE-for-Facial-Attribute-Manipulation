import matplotlib.pyplot as plt

def plot_img_and_recon(img, recon_img):
    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img.detach().cpu().permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(recon_img.detach().cpu().permute(1, 2, 0))
    plt.axis("off")

    plt.show()