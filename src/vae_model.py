import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=200, latent_dim=10):
        super(VAE, self).__init__()
        
        # Encoder: Compresses the IVS into latent factors [cite: 12]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: Reconstructs the full surface from latent vector z 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Softplus() # Ensures Implied Volatility is always positive
        )

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Flatten input: (Batch, 10, 20) -> (Batch, 200)
        x_flat = x.view(x.size(0), -1)
        
        h = self.encoder(x_flat)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        # Reshape back to (Batch, 10, 20)
        return recon_x.view(x.size(0), 10, 20), mu, logvar