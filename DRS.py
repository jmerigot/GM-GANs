import torch
import numpy as np

from model import Generator, Discriminator
from utils import load_model_D, load_model_G

def burn_in_discr(D, G, burn_in_samples=10000):
    """
    Burn-in phase to estimate the maximum density ratio.
    """
    max_ratio = float('-inf')
    for _ in range(burn_in_samples):
        z = torch.randn(1, G.input_size)  # Assuming G.input_size is the size of the latent vector
        fake_data = G(z)
        ratio = D(fake_data).item()
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio


def discr_rejection_sampling(D, G, max_ratio, num_samples=1000, epsilon=1e-6, gamma=1.0):
    """
    Discriminator Rejection Sampling to generate high-quality samples.
    """
    samples = []
    while len(samples) < num_samples:
        z = torch.randn(1, G.input_size)
        fake_data = G(z)
        logit = D(fake_data).item()

        # Equation 8 (modified version for simplicity)
        f_hat = logit - max_ratio - np.log(1 - np.exp(logit - max_ratio - epsilon)) - gamma
        p = 1 / (1 + np.exp(-f_hat))  # Sigmoid function

        # Acceptance check
        if np.random.uniform(0, 1) <= p:
            samples.append(fake_data.detach().numpy())
    
    return samples

################################################################################################


# Example usage
G = Generator()  # Replace with your generator model
D = Discriminator()  # Replace with your discriminator model

# Load your pre-trained models
G.load_state_dict(torch.load('path_to_generator_model.pt'))
D.load_state_dict(torch.load('path_to_discriminator_model.pt'))

G.eval()
D.eval()

# Burn-in phase
max_ratio = burn_in_discr(D, G)

# Generate high-quality samples
high_quality_samples = discr_rejection_sampling(D, G, max_ratio)

# Now `high_quality_samples` contains generated samples filtered by DRS
