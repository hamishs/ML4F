import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class PositionalEncoding(nn.Module):
	'''
	The positional encoding class is used in the encoder and decoder layers.
	It's role is to inject sequence order information into the data since self-attention
	mechanisms are permuatation equivariant. Naturally, this is not required in the static
	transformer since there is no concept of 'order' in a portfolio.'''

	def __init__(self, window, d_model):
		super().__init__()

		self.register_buffer('d_model', torch.tensor(d_model, dtype = torch.float64))

		pe = torch.zeros(window, d_model)
		for pos in range(window):
			for i in range(0, d_model, 2):
			  pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
				
			for i in range(1, d_model, 2):
			  pe[pos, i] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))             
				
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		return x * torch.sqrt(self.d_model) + self.pe[:,:x.size(1)]

def create_mask(seq_len):
	'''
	Create a mask to be used in the decoder.
	Returns a mask of shape (1, seq_len, seq_len)
	'''
	no_peak_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype('uint8')
	return torch.from_numpy(no_peak_mask)

def get_clones(module, N):
	'''
	This helper function is used to create copies of encoder and decoder layers.
	These copies of encoder/decoder layers are used to construct the
	complete stacked encoder/decoder modules.
	'''
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def scaled_dot_product_attention(k, q, v, mask = None):
	'''
	k : (batch, seq_len_k, heads, d)
	q : (batch, seq_len_q, heads, d)
	v : (batch, seq_len_v, heads, d)

	require seq_len_k == seq_len_v
	'''

	b, _, h, d = k.shape

	k = k.permute(0, 2, 3, 1) # (..., d, seq_len_k)
	q = q.permute(0, 2, 1, 3) # (..., seq_len_q, d)
	v = v.permute(0, 2, 1, 3) # (..., seq_len_v, d)

	scores = torch.matmul(q, k) # (..., seq_len_q, seq_len_k)
	dk = torch.tensor(d, dtype = torch.float32)
	scores = scores / torch.sqrt(dk)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	scores = F.softmax(scores, dim = -1)

	output = torch.matmul(scores, v) # (..., seq_len_q, d)
	output = output.permute(0, 2, 1, 3) # (batch, seq_len_q, heads, d)

	return output, scores # (batch, seq_len_q, heads, d)

class MultiHeadAttention(nn.Module):
	'''This is a Mult-Head self-attention class.'''
	def __init__(self, heads, d_model, dropout = 0.1):
		super().__init__()

		assert d_model % heads == 0
		
		self.h = heads
		self.d_model = d_model
		self.depth = d_model // heads

		self.dropout = nn.Dropout(dropout)
		
		self.q_linear = nn.Linear(d_model, d_model,bias=False)
		self.v_linear = nn.Linear(d_model, d_model,bias=False)
		self.k_linear = nn.Linear(d_model, d_model,bias=False)
		  
		self.o_linear = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, mask = None):

		b = q.shape[0]

		k = self.k_linear(k).view(b, -1, self.h, self.depth)
		q = self.q_linear(q).view(b, -1, self.h, self.depth)
		v = self.v_linear(v).view(b, -1, self.h, self.depth)

		output, attn_weights = scaled_dot_product_attention(k, q, v, mask = mask)

		output = output.reshape(b, -1, self.d_model)
		output = self.o_linear(output) # (batch, seq_len, d_model)

		return output, attn_weights

class FeedForward(nn.Module):
	'''This is a pointwise feedforward network.'''
	def __init__(self, d_model, dff, dropout = 0.1):
		super().__init__()
		
		self.ff = nn.Sequential(
			nn.Linear(d_model, dff),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dff, d_model))
	
	def forward(self, x):
		x = self.ff(x)
		return x 

class EncoderLayer(nn.Module):
	'''Encoder layer class.'''
	def __init__(self, heads, d_model, dff, dropout = 0.1):
		super().__init__()

		self.norm_1 = nn.LayerNorm(d_model)
		self.norm_2 = nn.LayerNorm(d_model)

		self.attn = MultiHeadAttention(heads, d_model, dropout = dropout)
		self.ff = FeedForward(d_model, dff)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
	
	def forward(self, x):
		attn_out, _ = self.attn(x, x, x)
		attn_out = self.dropout_1(attn_out)
		x = self.norm_1(x + attn_out)

		ffn_out = self.ff(x)
		x = self.norm_2(x + ffn_out)

		return x

class Encoder(nn.Module):
	'''Stacked encoder layers.'''
	def __init__(self, N, pe_window, heads, inp_dim, d_model, dff, dropout):
		super().__init__()

		self.N = N
		self.embedding = nn.Linear(inp_dim, d_model)
		self.pe = PositionalEncoding(pe_window, d_model)
		self.dynamiclayers = get_clones(EncoderLayer(heads, d_model, dff, dropout = dropout), N)

	def forward(self, x):
		# x (batch, seq_len, inp_dim)

		x = self.embedding(x) # (batch, seq_len, d_model)
		
		x = self.pe(x) # (batch, seq_len, d_model)

		for i in range(self.N):
		  x = self.dynamiclayers[i](x) # (batch, seq_len, d_model)

		return x # (batch, seq_len, d_model)

class DecoderLayer(nn.Module):
	'''Decoder Layer class'''
	def __init__(self, heads, d_model, dff, dropout = 0.1):
		super().__init__()

		self.norm_1 = nn.LayerNorm(d_model)
		self.norm_2 = nn.LayerNorm(d_model)
		self.norm_3 = nn.LayerNorm(d_model)

		self.attn_1 = MultiHeadAttention(heads, d_model, dropout = dropout)
		self.attn_2 = MultiHeadAttention(heads, d_model, dropout = dropout)
		self.ff = FeedForward(d_model, dff)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)

	def forward(self, x, enc_out, mask = None):
		# x (batch, seq_len, d_model)
		# enc_out (batch, enc_seq_len, d_model)

		attn_1_out, _ = self.attn_1(x, x, x, mask = mask)
		attn_1_out = self.dropout_1(attn_1_out)
		x = self.norm_1(x + attn_1_out) # (batch, seq_len, d_model)

		attn_2_out, attn_weights = self.attn_2(x, enc_out, enc_out)
		attn_2_out = self.dropout_2(attn_2_out)
		x = self.norm_2(x + attn_2_out) # (batch, seq_len, d_model)

		ffn_out = self.dropout_3(self.ff(x))
		x = self.norm_3(x + ffn_out) # (batch, seq_len, d_model)

		return x, attn_weights # (batch, seq_len, d_model)

class Decoder(nn.Module):
	'''Stacked decoder layers.'''
	def __init__(self, N, pe_window, heads, inp_dim, d_model, dff, dropout = 0.1):
		super().__init__()

		self.N = N
		self.embedding = nn.Linear(inp_dim, d_model)
		self.pe = PositionalEncoding(pe_window, d_model)
		self.decoderlayers = get_clones(DecoderLayer(heads, d_model, dff, dropout = dropout), N)

	def forward(self, x, enc_out, mask = None):
		# x (batch, seq_len, inp_dim)
		# enc_out (batch, enc_seq_len, d_model)

		attention_weights = {}

		x = self.embedding(x) # (batch, seq_len, d_model)

		x = self.pe(x) # (batch, seq_len, d_model)
		
		for i in range(self.N):
			x, attn_weights = self.decoderlayers[i](x, enc_out, mask = mask) # (batch, seq_len, d_model)
			attention_weights[i] = attn_weights
		
		return x, attention_weights # (batch, seq_len, d_model)

class Ml4fTransformer(nn.Module):
	'''
	Main transformer class.
	experiment : selects sigmoid final activation if 'movement' else linear
	inp_dim_e : number of dimensions of encoder input
	inp_dim_d : number of dimensions of decoder input
	d_model : model embedding dimension
	dff : hidden dimension of feed forward network
	N_e : number of encoder layers
	N_d : number of decoder layers
	heads : number of heads
	'''
	def __init__(self, inp_dim_e, inp_dim_d, experiment = 'return', d_model = 20,
		dff = 80, N_e = 1, N_d = 1, heads = 4, dropout = 0.1, pe_window = 100):
		super().__init__()

		assert d_model % heads == 0
		
		self.encoder = Encoder(N_e, pe_window, heads, inp_dim_e, d_model, dff, dropout = dropout)
		
		self.decoder = Decoder(N_d, pe_window, heads, inp_dim_d, d_model, dff, dropout = dropout)
		
		if experiment == 'hit':
			self.map = nn.Sequential(
				nn.Linear(d_model, inp_dim_d),
				nn.Sigmoid())
		else:
			self.map = nn.Linear(d_model, inp_dim_d)
	
	def forward(self, x, y, mask = None):
		'''
		x (batch, in_seq_len, inp_dim_e)
		y (batch, tar_seq_len, inp_dim_d)
		'''

		enc_out = self.encoder(x) # (batch, in_seq_len, d_model)

		dec_out, attention_weights = self.decoder(y, enc_out, mask = mask) # (batch, tar_seq_len, d_model)

		final = self.map(dec_out) # (batch, tar_seq_len, 1)

		return final, attention_weights

class LrSchedule:
	'''
	Implements the learning rate schedule from "Attention is all you need".
	Learning rate given by (1/sqrt(d_model)) * min(steps^-0.5, steps * warmup_steps ^-1.5).

	Init with a pytorch.optim optimizer then call as usual in the training loop.
	'''
	def __init__(self, optimizer, warmup_steps, d_model, scale = 1.0):
		self.optimizer_ = optimizer 
		self.warmup_steps = warmup_steps
		self.d_model = d_model
		self.scale = scale
		self.step_ = 1

	def zero_grad(self):
		self.optimizer_.zero_grad()

	def get_lr(self, step):
		min_ = np.minimum(step ** -0.5, step * (self.warmup_steps ** -1.5))
		return (self.d_model ** -0.5) * min_

	def update_lr(self):
		lr = self.scale * self.get_lr(self.step_)
		self.step_ += 1

		for param_group in self.optimizer_.param_groups:
			param_group['lr'] = lr

	def step(self):
		self.update_lr()
		self.optimizer_.step()

	def reset_lr(self, lr = 1e-3):
		for param_group in self.optimizer_.param_groups:
			param_group['lr'] = lr

if __name__ == '__main__':

	model = Ml4fTransformer(5, 1)

	x = torch.randn(4, 15, 5)
	y = torch.randn(4, 5, 1)
	mask = create_mask(y.shape[1])

	out, attn_weights = model(x, y, mask)

	print(out)
	print(out.shape)
	print(attn_weights)


