import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class VRAE(nn.Module):

    def __init__(self, input_dim=5, hidden_dim=50, latent_dim=20, num_layers_encoder=3, num_layers_decoder=2):
        super(VRAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_encoder,
            batch_first=True,
            bidirectional=True
        )

        encoder_output_dim = 2 * hidden_dim

        self.hidden_to_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
  
        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_decoder,
            batch_first=True
        )

        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)
    
    
    def encode(self, x, lengths):

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        _, hidden = self.encoder(packed)

        hidden_forward = hidden[-2]  
        hidden_backward = hidden[-1] 
        hidden_cat = torch.cat([hidden_forward, hidden_backward], dim=1)

        mean = self.hidden_to_mean(hidden_cat)
        logvar = self.hidden_to_logvar(hidden_cat)
        
        return mean, logvar
    
    
    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        
        return z
    
    
    def decode(self, z, target_length):

        batch_size = z.shape[0]
        hidden = self.latent_to_hidden(z)  
        hidden = torch.tanh(hidden)
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=z.device)

        outputs = []
        
        for t in range(target_length):

            out, hidden = self.decoder(decoder_input, hidden)
         
            prediction = self.hidden_to_output(out)
            outputs.append(prediction)
            
            decoder_input = prediction
        
        reconstruction = torch.cat(outputs, dim=1)
        
        return reconstruction
    
    
    def forward(self, x, lengths):

        mean, logvar = self.encode(x, lengths)
        z = self.reparameterize(mean, logvar)
        max_length = x.shape[1]
        reconstruction = self.decode(z, max_length)
        
        return reconstruction, mean, logvar
    
def vae_loss(reconstruction, target, mu, logvar, beta=1.0):

    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.size(0)  # Normalize by batch

    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss