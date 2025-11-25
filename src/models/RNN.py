# initialize the RNN model
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):  
    # TODO: 
    #   what is the size of the input/output?

    def __init__(self, input_size, output_size, num_layers=1, hidden_size=128, dropout = 0.1, prediction_horizon=30):
        super(RNNModel, self).__init__()

        self.input_dim = input_size
        self.output_dim = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.prediction_horizon = prediction_horizon

        # Encoder reads the past sequence (5 features for the past 30 mins)
        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder generates the future sequence
        # at each step, it takes the previous output as input
        self.decoder = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Map decoder hidden state -> predicted output
        self.output_head = nn.Linear(self.hidden_size, self.output_dim)

        # Map last encoder input (full features) -> initial "output space" vectore for decoder
        # This gives the decoder something meaningful to start from at t=0
        self.init_output = nn.Linear(self.input_dim, self.output_dim)


    def forward(self, past_seq, target_future, teacher_forcing_ratio=0.1) -> torch.Tensor:
        # Now has 100% teacher forcing ratio - always uses true previous output during training

        batch_size = past_seq.size(0)
        device = past_seq.device

        # -- ENCODER: read the past --
        _, hidden = self.encoder(past_seq)  # hidden: (num_layers, batch, hidden_size)

        # -- DECODER --
 
        # Training mode: we know the ground truth future (30 min)
        T_out = target_future.size(1)

        # First decder input is all zeros
        # Then at each step we feed the true previous future
        first_input = torch.zeros(batch_size, 1, self.output_dim).to(device)
        # Shift future target by one step to the right
        dec_input = torch.cat([first_input, target_future[:, :-1, :]], dim=1)  # (B, T_out, output_dim)

        dec_out, _ = self.decoder(dec_input, hidden)  # dec_out: (B, T_out, hidden_size)
        preds = self.output_head(dec_out)  # (B, T_out, output_dim
       
        # -- DECODER LOOP --
        for t in range(T_out):
            # Prepare input
            # Expects sequence of length 1 at each step: (B, 1, output_dim)
            dec_input = prev_output.unsqueeze(1)  

            # Pass through decoder
            dec_output, hidden = self.decoder(dec_input, hidden)  # dec_output: (B, 1, hidden_size)

            # Project to output space
            step_output = self.output_head(dec_output[:, -1, :])
            outputs[:, t, :] = step_output

            # Decide next input: teacher forcing!!!
            # TODO: implement teacher forcing
            if target_future is not None and (t + 1 < T_out):
                prev_output = target_future[:, t, :]
            else:
                # at the last step of the prediction horizon, no next input is needed
                prev_output = step_output.detach()

        return outputs