import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        enc_out_dim = hidden_size * self.num_directions


        # map final hidden state -> latent mean and logvar
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: z, mu, logvar  with shape (batch, latent_dim)
        """
        output, h_n = self.gru(x)         # h_T: (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(), -1)
        h_last = h_n[-1]                       # (num_directions, batch, hidden)
        h_last = h_last.transpose(0, 1)        # (batch, num_directions, hidden)
        h_last = h_last.reshape(x.size(0), -1) # (batch, hidden * num_directions)

        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Recurrent layer
        self.gru = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # Output layer
        self.output_layer = nn.Linear(in_features=hidden_size,
                            out_features=input_size )

    def forward(self, z, seq_len):
      """
      z: (batch, latent_dim)
      seq_len: int (length of sequence to reconstruct)
      returns: x_hat: (batch, seq_len, input_size)
      """
      batch_size = z.size(0)

      # Initial hidden state from latent vector
      h0 = self.z_to_h0(z)                         # (batch, hidden_size * num_layers)
      h0 = h0.view(self.num_layers, batch_size, self.hidden_size).contiguous()

      # Start decoder with all zeros as inputs
      # (You can switch to teacher forcing later if you want)
      dec_inputs = torch.zeros(
          batch_size,
          seq_len,
          self.input_size,
          device=z.device,
          dtype=z.dtype,
      )

      dec_outputs, _ = self.gru(dec_inputs, h0)    # (batch, seq_len, hidden_size)
      x_hat = self.output_layer(dec_outputs)       # (batch, seq_len, input_size)

      return x_hat

class VRAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, bidirectional=False):
        super(VRAE, self).__init__()

        self.encoder = RNNEncoder(input_size=input_dim,
            hidden_size=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            bidirectional=bidirectional)
        
        self.decoder = RNNDecoder(input_size=input_dim,
            hidden_size=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers)

    def encode_mu(self, x, lengths=None):
        """
        x: (batch, seq_len, input_dim)
        lengths: optional sequence lengths

        Returns:
            mu: (batch, latent_dim) – deterministic latent embedding
        """
        self.eval()
        with torch.no_grad():
            _, mu, _ = self.encoder(x, lengths)
        return mu
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z, seq_len)

        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

    @staticmethod
    def loss_function(x, x_hat, mu, logvar, beta=1.0):
        """
        Standard VAE loss: reconstruction + KL
        x, x_hat: (batch, seq_len, input_dim)
        """
        # Reconstruction loss (MSE over all time steps & features)
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # KL divergence between q(z|x) and N(0, I)
        # Average over batch (and optionally over time if you wanted per-timestep latents)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / x.size(0)  # normalize by batch size

        return recon_loss + beta * kld, recon_loss, kld
