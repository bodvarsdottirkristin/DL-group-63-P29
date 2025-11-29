import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class TrajectoryPredictor(nn.Module):
    """
    Sequence-to-sequence trajectory predictor.
    input: past sequence x
    output future sequence y

    input_dim:    dimension of each input vector (5 features)
    hidden_dim:   dimension of hidden states in encoder/decoder GRUs
    output_dim:   dimension of each output vector (5 features)
    num_layers_encoder: number of layers in encoder GRU
    num_layers_decoder: number of layers in decoder GRU
    attn_dim:     dimension of attention mechanism

    """
    def __init__(
        self,
        input_dim=5,
        hidden_dim=20,
        output_dim=5,
        num_layers_encoder=2,
        num_layers_decoder=2,
        attn_dim=32,
    ):
        super(TrajectoryPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.attn_dim = attn_dim

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

        # Bahdanau-style attention: score(h_dec_prev, h_enc_i) = v^T tanh(W[h_dec_prev; h_enc_i])
        self.attn_mlp = nn.Linear(hidden_dim + encoder_output_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        # Input at each time step: concat([previous prediction y_{t-1}, weighted encoder state w_t])
        decoder_input_dim = output_dim + encoder_output_dim
        # Decoder (unidirectional GRU over future window)
        # Decoder input at each time step is previous predicted state (dim = output_dim)
        self.decoder = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_decoder,
            batch_first=True,
        )

        # Map decoder hidden state to output state (same dim as input state)
        # Output linear: uses [h_t, y_{t-1}, w_t]
        out_in_dim = hidden_dim + output_dim + encoder_output_dim
        self.hidden_to_output = nn.Linear(out_in_dim, output_dim)

    def encode(self, x):
        """
        Encode a past trajectory sequence.

        x:       (batch, T_in, input_dim)

        Returns:
            enc_outputs: (batch, T_in, 2*hidden_dim)  # h_0..h_L for attention
            enc_summary: (batch, 2*hidden_dim)        # final concat fwd/bwd
        """

        enc_outputs, hidden = self.encoder(x)
        # enc_outputs: (num_layers*2, B, H) if batch_first=False; but we used batch_first=True,
        # so enc_outputs: (B, T_in, 2*hidden_dim)

        # hidden: (num_layers_encoder * num_directions, batch, hidden_dim)
        # num_directions = 2 (bidirectional)

        # Last encoder layer's forward and backward states
        hidden_forward = hidden[-2]   # (batch, hidden_dim)
        hidden_backward = hidden[-1]  # (batch, hidden_dim)
        enc_summary = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, 2*hidden_dim)

        return enc_outputs,  enc_summary # encoder summary vector

    # Attention: compute w_t given previous decoder hidden and encoder outputs
    def compute_attention(self, h_dec_prev, enc_outputs):

        """
        h_dec_prev:  (batch, hidden_dim)           # previous decoder top-layer hidden
        enc_outputs: (batch, T_in, 2*hidden_dim)   # encoder outputs at each time

        Returns:
            w_t: (batch, 2*hidden_dim)            # weighted sum of encoder states
            a_t: (batch, T_in)                    # attention weights
        """

        batch_size, T_in, enc_dim = enc_outputs.shape  # enc_dim = 2*hidden_dim

        # Repeat h_dec_prev across time steps: (B, T_in, hidden_dim)
        h_dec_expanded = h_dec_prev.unsqueeze(1).expand(-1, T_in, -1)
        # Concatenate along feature axis: (B, T_in, hidden_dim + 2*hidden_dim)
        attn_input = torch.cat([h_dec_expanded, enc_outputs], dim=2)

        # Compute scores: (B, T_in, attn_dim) -> (B, T_in, 1) -> (B, T_in)
        energy = torch.tanh(self.attn_mlp(attn_input))       # (B, T_in, attn_dim)
        scores = self.attn_v(energy).squeeze(-1)             # (B, T_in)

        # Normalize with softmax over time
        a_t = F.softmax(scores, dim=1)                       # (B, T_in)

        # Weighted sum of encoder outputs
        w_t = torch.bmm(a_t.unsqueeze(1), enc_outputs)       # (B, 1, 2*hidden_dim)
        w_t = w_t.squeeze(1)                                 # (B, 2*hidden_dim)

        return w_t, a_t

    def decode(self, enc_outputs, enc_summary, target_length, y_start, targets=None, teacher_forcing_ratio=0.5):
        """
        Decode future trajectory given encoder outputs + summary.

        enc_outputs:   (batch, T_in, 2*hidden_dim)
        enc_summary:   (batch, 2*hidden_dim)  from encode(...)
        target_length: int, number of future steps to predict (T_out)
        y_start:       (batch, output_dim) initial decoder input (last known state)
        targets:       (batch, T_out, output_dim) or None
        """
         
        batch_size = enc_outputs.size(0)

        # Map encoder summary to initial decoder hidden
        hidden = self.hidden_enc_to_dec(enc_summary)      # (batch, hidden_dim)
        hidden = torch.tanh(hidden)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers_decoder, 1, 1)
        # hidden: (num_layers_decoder, batch, hidden_dim)

        y_prev = y_start  # (B, output_dim)

        outputs = []

        for t in range(target_length):
            # top-layer decoder hidden from previous step
            h_dec_prev = hidden[-1]  # (B, hidden_dim)

            # 1) Attention: get weighted encoder summary w_t
            w_t, a_t = self.compute_attention(h_dec_prev, enc_outputs)  # (B, 2*H), (B, T_in)

            # 2) Decoder GRU input is concat([y_{t-1}, w_t])
            dec_input = torch.cat([y_prev, w_t], dim=1)  # (B, output_dim + 2*H)
            dec_input = dec_input.unsqueeze(1)           # (B, 1, ...)
            dec_out, hidden = self.decoder(dec_input, hidden)  # dec_out: (B, 1, H)
            h_t = dec_out.squeeze(1)                    # (B, H)

            # 3) Output layer uses [h_t, y_{t-1}, w_t]
            out_in = torch.cat([h_t, y_prev, w_t], dim=1)   # (B, H + output_dim + 2*H)
            y_t = self.hidden_to_output(out_in)             # (B, output_dim)
            outputs.append(y_t.unsqueeze(1))

            # 4) Next input: teacher forcing or own prediction
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                y_prev = targets[:, t, :]   # (B, output_dim)
            else:
                y_prev = y_t                # (B, output_dim)

        preds = torch.cat(outputs, dim=1)  # (B, T_out, output_dim)
        return preds

    def forward(self, x, target_length, targets=None, teacher_forcing_ratio=0.5):
        """
        x:       (batch, T_in, input_dim)       past window
        lengths: (batch,)                       past lengths (for packing)
        target_length: int                      T_out
        targets: (batch, T_out, output_dim) or None

        Returns:
            preds: (batch, T_out, output_dim)
        """
        # 1) Encode past sequence
        enc_outputs, enc_summary = self.encode(x)  # (batch, T_in, 2*hidden_dim), (batch, 2*hidden_dim)

        # 2) Initial decoder input: last observed state from x
        # assumes output_dim <= input_dim and first output_dim dims of x are the state we predict
        y_start = x[:, -1, : self.output_dim]  # (batch, output_dim)

        # 3) Decode future sequence
        preds = self.decode(
            enc_outputs=enc_outputs,
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