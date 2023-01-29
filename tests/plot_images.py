import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruct(x, x_hat):
    # Select 3 random indices
    indices = np.random.randint(0, 64, size=3)

    # Plot the selected images
    for i, index in enumerate(indices):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(x[index, 0, :, :], cmap='gray')
        plt.title("Original Image")

        plt.subplot(3, 2, 2*i+2)
        plt.imshow(x_hat[index, 0, :, :], cmap='gray')
        plt.title("Reconstructed Image")

    plt.show()