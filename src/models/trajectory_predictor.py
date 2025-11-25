import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryPredictor(nn.Module):
    """
    Sequence-to-sequence trajectory predictor.

    past sequence x -> future sequence y
    """
    def __init__(
        self,
        input_dim=5,
        hidden_dim=20,
        output_dim=5,
        num_layers_encoder=2,
        num_layers_decoder=2,
    ):
        super(TrajectoryPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder

        # Encoder (bidirectional GRU over past window)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_encoder,
            batch_first=True,
            bidirectional=True,
        )

        encoder_output_dim = 2 * hidden_dim  # because bidirectional

        # Map encoder final hidden (concat fwd/bwd) to decoder initial hidden
        self.hidden_enc_to_dec = nn.Linear(encoder_output_dim, hidden_dim)

        # Decoder (unidirectional GRU over future window)
        # Decoder input at each time step is previous predicted state (dim = output_dim)
        self.decoder = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_decoder,
            batch_first=True,
        )

        # Map decoder hidden state to output state (same dim as input state)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

    def encode(self, x, lengths):
        """
        Encode a past trajectory sequence.

        x:       (batch, T_in, input_dim)
        lengths: (batch,) length of each sequence (for packing)
        """
        # Pack for variable-length sequences (same pattern as in VRAE)
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        _, hidden = self.encoder(packed)
        # hidden: (num_layers_encoder * num_directions, batch, hidden_dim)
        # num_directions = 2 (bidirectional)

        # Last encoder layer's forward and backward states
        hidden_forward = hidden[-2]   # (batch, hidden_dim)
        hidden_backward = hidden[-1]  # (batch, hidden_dim)
        hidden_cat = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, 2*hidden_dim)

        return hidden_cat  # encoder summary vector

    def decode(self, enc_summary, target_length, y_start, targets=None, teacher_forcing_ratio=0.5):
        """
        Decode future trajectory given encoder summary.

        enc_summary:    (batch, 2*hidden_dim)  from encode(...)
        target_length:  int, number of future steps to predict (T_out)
        y_start:        (batch, output_dim) initial decoder input (e.g. last known state)
        targets:        (batch, T_out, output_dim) or None
        """
        batch_size = enc_summary.shape[0]

        # Map encoder summary to initial decoder hidden
        hidden = self.hidden_enc_to_dec(enc_summary)      # (batch, hidden_dim)
        hidden = torch.tanh(hidden)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers_decoder, 1, 1)
        # hidden: (num_layers_decoder, batch, hidden_dim)

        decoder_input = y_start.unsqueeze(1)  # (batch, 1, output_dim)

        outputs = []

        for t in range(target_length):
            out, hidden = self.decoder(decoder_input, hidden)  # out: (batch, 1, hidden_dim)
            prediction = self.hidden_to_output(out)            # (batch, 1, output_dim)
            outputs.append(prediction)

            # Decide next input with teacher forcing
            if (
                targets is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                # Use ground truth at current step t
                next_input = targets[:, t, 🙂  # (batch, output_dim)
            else:
                # Use model's own prediction
                next_input = prediction.squeeze(1)  # (batch, output_dim)

            decoder_input = next_input.unsqueeze(1)  # (batch, 1, output_dim)

        preds = torch.cat(outputs, dim=1)  # (batch, target_length, output_dim)
        return preds

    def forward(self, x, lengths, target_length, targets=None, teacher_forcing_ratio=0.5):
        """
        x:       (batch, T_in, input_dim)       past window
        lengths: (batch,)                       past lengths (for packing)
        target_length: int                      T_out
        targets: (batch, T_out, output_dim) or None

        Returns:
            preds: (batch, T_out, output_dim)
        """
        # 1) Encode past sequence
        enc_summary = self.encode(x, lengths)  # (batch, 2*hidden_dim)

        # 2) Initial decoder input: last observed state from x
        # assumes output_dim <= input_dim and first output_dim dims of x are the state we predict
        y_start = x[:, -1, : self.output_dim]  # (batch, output_dim)

        # 3) Decode future sequence
        preds = self.decode(
            enc_summary=enc_summary,
            target_length=target_length,
            y_start=y_start,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return preds


def trajectory_loss(preds, targets):
    """
    Simple MSE loss for trajectory prediction.

    preds:   (batch, T_out, output_dim)
    targets: (batch, T_out, output_dim)

    If we only care about the position error (p1,p2),
    we can slice before calling this:
        trajectory_loss(preds[..., :2], targets[..., :2])
    """
    return F.mse_loss(preds, targets, reduction="mean")