from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch

class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers_encoder=3, num_layers_decoder=2, dropout=0.1):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        
        # Bidirectional stacked encoder
        self.encoder = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers_encoder,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers_encoder > 1 else 0
        )
        
        # Compress to latent
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)  # *2 for bidirectional
        self.bn_latent = nn.LayerNorm(latent_dim)  # More stable for small/variable batches
        
        # Expand from latent to decoder initial state
        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers_decoder)
        
        # Stacked decoder
        self.decoder = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers_decoder,
            batch_first=True,
            dropout=dropout if num_layers_decoder > 1 else 0
        )
        
        # Output layer
        self.fc_output = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x, lengths):
        """Encode trajectory to latent vector"""
        # Pack padded sequence
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Encode
        _, hidden = self.encoder(packed)
        
        # Concatenate forward and backward from top layer
        # hidden shape: [num_layers*2, batch, hidden_dim]
        forward_hidden = hidden[-2]  # Top layer forward
        backward_hidden = hidden[-1]  # Top layer backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to latent space
        z = self.fc_latent(combined)
        z = self.bn_latent(z)  # Normalize per-sample
        z = torch.tanh(z)      # Bounded latent for stability
        
        return z
    
    def decode(self, z, target_seq, lengths, teacher_forcing_ratio=0.0):
        """Decode from latent vector to trajectory"""
        batch_size = z.size(0)
        max_length = target_seq.size(1)
        
        # Initialize decoder hidden state from z
        hidden = self.fc_z_to_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers_decoder, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # [num_layers, batch, hidden]
        
        # Start token (use first timestep of target)
        decoder_input = target_seq[:, 0, :].unsqueeze(1)  # [batch, 1, input_dim]
        
        outputs = []
        
        for t in range(max_length):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            output = self.fc_output(output)  # [batch, 1, input_dim]
            
            outputs.append(output)
            
            # Teacher forcing
            if t < max_length - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    decoder_input = target_seq[:, t+1, :].unsqueeze(1)
                else:
                    decoder_input = output
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # [batch, seq_len, input_dim]
        
        return outputs
    
    def forward(self, x, lengths, teacher_forcing_ratio=0.0):
        """Full forward pass"""
        z = self.encode(x, lengths)
        reconstruction = self.decode(z, x, lengths, teacher_forcing_ratio)
        return reconstruction, z



def pad_trajectories(batch):
    lengths = torch.tensor([len(traj) for traj in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)

    lengths, perm_idx = lengths.sort(descending=True)
    padded = padded[perm_idx]

    return padded, lengths


def masked_mse(reconstruction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Compute MSE only over valid (unpadded) timesteps.

    Args:
        reconstruction: [batch, seq_len, feature]
        target: [batch, seq_len, feature]
        lengths: [batch] lengths per sequence

    Returns:
        Scalar tensor loss averaged over valid elements.
    """
    # Create mask over timesteps: [batch, seq_len]
    device = reconstruction.device
    seq_len = reconstruction.size(1)
    time_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    mask = (time_ids < lengths.unsqueeze(1))  # [batch, seq_len] boolean
    # Expand to feature dim: [batch, seq_len, feature]
    mask = mask.unsqueeze(-1)
    # Squared error per element
    se = (reconstruction - target) ** 2
    # Apply mask and average over valid elements only
    se_masked = se * mask
    # Avoid division by zero if a length is 0 (shouldn't happen, but safe)
    valid_count = mask.sum().clamp_min(1)
    loss = se_masked.sum() / valid_count
    return loss

def train_RAE(
        train_dataset,
        val_dataset,
        device,
        feature_size,
        encoder_layers=3, 
        decoder_layers=2, 
        latent_dim=8, 
        hidden_size=48, 
        learning_rate=0.001, 
        batch_size=256, 
        max_epochs=50, 
        teacher_forcing_ratio=0.9,
        dropout=0.1,
        patience=10, 
        min_delta=0.0001
    ):
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_trajectories
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_trajectories
    )
    
    # Initialize model
    model = RecurrentAutoencoder(
        input_dim=feature_size,
        hidden_dim=hidden_size,
        latent_dim=latent_dim,
        num_layers_encoder=encoder_layers,
        num_layers_decoder=decoder_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Use masked loss instead of naive MSE over padded timesteps
    criterion = masked_mse
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss tracking
    train_losses, val_losses = [], []
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training RAE: latent_dim={latent_dim}, hidden={hidden_size}, "
          f"enc_layers={encoder_layers}, dec_layers={decoder_layers}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*70)
    
    for epoch in range(max_epochs):
        
        #
        model.train()
        train_loss = 0
        
        for batch_data, lengths in train_loader:
            batch_data = batch_data.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            reconstruction, z = model(batch_data, lengths, teacher_forcing_ratio)
            
            # Compute masked reconstruction loss
            loss = criterion(reconstruction, batch_data, lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        # Collect latent representations for diagnostics
        all_latents = []
        
        with torch.no_grad():
            for batch_data, lengths in val_loader:
                batch_data = batch_data.to(device)
                lengths = lengths.to(device)
                
                reconstruction, z = model(batch_data, lengths, teacher_forcing_ratio=0.0)
                loss = criterion(reconstruction, batch_data, lengths)
                
                val_loss += loss.item()
                all_latents.append(z.cpu())
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Diagnostics on latent space
        if epoch == 0 or epoch == max_epochs - 1 or (epoch + 1) % 5 == 0:
            all_latents = torch.cat(all_latents, dim=0).numpy()
            
            latent_mean = all_latents.mean(axis=0)
            latent_std = all_latents.std(axis=0)
            latent_overall_std = all_latents.std()
            
            # Per-dimension variance
            var_per_dim = all_latents.var(axis=0)
            active_dims = (var_per_dim > 0.01).sum()
            
            print(f"\n{'='*70}")
            print(f"DIAGNOSTICS - Epoch {epoch+1}")
            print(f"{'='*70}")
            print(f"Latent space statistics:")
            print(f"  Mean (avg across dims): {latent_mean.mean():.4f}")
            print(f"  Std (overall): {latent_overall_std:.4f} (should be > 0.3)")
            print(f"  Active dimensions: {active_dims}/{latent_dim} (var > 0.01)")
            print(f"\nPer-dimension statistics:")
            for i in range(latent_dim):
                status = "✓" if var_per_dim[i] > 0.01 else "⚠️  LOW"
                print(f"  Dim {i}: mean={latent_mean[i]:+.3f}, "
                      f"std={latent_std[i]:.3f}, var={var_per_dim[i]:.4f} {status}")
            print(f"{'='*70}\n")
        

        tf_str = f"TF={teacher_forcing_ratio:.1f}" if teacher_forcing_ratio > 0 else "NoTF"
        
        # Early stopping check
        if patience is not None:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"Epoch {epoch+1}/{max_epochs} [{tf_str}] - "
                      f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} ✓")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}/{max_epochs} [{tf_str}] - "
                      f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} "
                      f"(patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{max_epochs} [{tf_str}] - "
                  f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
    
    # Load best model
    if patience is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with val_loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses