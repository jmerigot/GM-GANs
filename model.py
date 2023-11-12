import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Latent_Generator(torch.nn.Module):
    def __init__(self, config):
        super(Latent_Generator, self).__init__() 
        self.config_latent = config["latent"]
        self.config_training = config["training"]
        self.n_gaussian = self.config_latent["n_gaussian"]
        self.dim = self.config_latent["dim"]
        self.law = self.config_latent["law"]
        self.learn_type = self.config_latent["learn_type"]
        self.c = self.config_latent["c"]
        self.sigma = self.config_latent["sigma"]
        self.covar_deg = self.config_latent["covar_type"]

        if self.law == "GM":
            self.categorical = Categorical(torch.tensor([1 / self.n_gaussian for _ in range(self.n_gaussian)]))
            if self.learn_type ==  "dynamic":
                self.categorical = Categorical(torch.tensor([1 / self.n_gaussian for _ in range(self.n_gaussian)]))
                    # self.alphas = torch.tensor(
                    #     [1 / self.n_gaussian for _ in range(self.n_gaussian)])
                self.mu = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.randn(self.dim))
                    for k in range(self.n_gaussian)])
                if self.covar_deg == "1":
                    self.sigma = torch.nn.Parameter(torch.randn(1).cuda())
                    self.A = [torch.eye(self.dim).cuda() * self.sigma
                              for k in range(self.n_gaussian)]
                elif self.covar_deg == "diag":
                    self.sigma = torch.nn.Parameter(torch.randn(self.dim))
                    self.A = [torch.eye(self.sigma) 
                              for k in range(self.n_gaussian)]
                elif self.covar_deg == "full":
                    self.A = torch.nn.ParameterList(
                        [torch.nn.Parameter(torch.randn((self.dim, self.dim)))
                         for k in range(self.n_gaussian)])
            
            if self.learn_type == "static":
                self.mu = [torch.distributions.Uniform(-self.c, self.c).sample((self.dim,)).cuda() for _ in range(self.n_gaussian)]
                self.A = [torch.eye(self.dim).cuda() * self.sigma for _ in range(self.n_gaussian)]

    
    def forward(self, batch_size):
        print("===============================================================")
        print(f"Received batch size: {batch_size}")

        if self.law == "GM":
            k = self.categorical.sample((batch_size,)).cuda() # Sample k for each item in the batch
            epsilon = torch.randn(batch_size, self.dim).cuda() # Sample epsilon for each item in the batch
            print(f"Shape of k: {k.shape}")
            print(f"Shape of epsilon: {epsilon.shape}")
            mu_k = torch.index_select(torch.stack(list(self.mu)), 0, k).cuda()
            A_k = torch.index_select(torch.stack(list(self.A)), 0, k).cuda()
            print(f"Shape of mu_k: {mu_k.shape}")
            print(f"Shape of A_k: {A_k.shape}")
    
            z = torch.bmm(A_k, epsilon.unsqueeze(-1)).squeeze() + mu_k
            print(f"Shape of z (latent vector): {z.shape}")
        
        elif self.law == "vanilla":
            print("Matteo", batch_size)
            z = torch.randn(batch_size, self.dim).cuda()
            print(f"Shape of z (vanilla): {z.shape}")

        print(f"Shape of z (vanilla) before return: {z.shape}")
        return z


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
