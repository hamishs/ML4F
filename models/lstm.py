import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
	''' LSTM Encoder - Decoder. Encodes the input sequence
	with a multi-layer LSTM. The hidden state and output
	of the encoder is used to initialise the decoder. The decoder
	is then used to generate predictions for the next element
	of the sequence.'''
	
	def __init__(self, inp_dim_e, inp_dim_d, experiment = 'return',
		d_model = 20, N = 1, dropout = 0.1):
		super(LSTM, self).__init__()
		
		self.encoder = nn.LSTM(input_size = inp_dim_e,
							   hidden_size = d_model,
							   num_layers = N,
							   dropout = dropout,
							   batch_first = True)
		
		self.decoder = nn.LSTM(input_size = inp_dim_d,
							   hidden_size = d_model,
							   num_layers = N,
							   dropout = dropout,
							   batch_first = True)
		
		if experiment == 'hit':
			self.map = nn.Sequential(
				nn.Linear(d_model, inp_dim_d),
				nn.Sigmoid())
		else:
			self.map = nn.Sequential(
				nn.Linear(d_model, inp_dim_d))
		
	def forward(self, x, y):
		'''
		x : (batch, len_x, inp_dim_e)
		y : (batch, len_y, inp_dim_d)
		'''
		
		# encode
		_, (h, c) = self.encoder(x)
		
		# decode
		y, _ = self.decoder(y, (h, c)) # (batch, seq_len, d_model)
		y = self.map(y) # (batch, seq_len, 1)
		
		return y

if __name__ == '__main__':

	model = LSTM(1, 1, N = 1)

	x = torch.tensor([0, 1, 2, 2], dtype = torch.float32).reshape(1, -1, 1)
	y = torch.tensor([1, 2, 2], dtype = torch.float32).reshape(1, -1, 1)

	out = model(x, y)

	print(out)

