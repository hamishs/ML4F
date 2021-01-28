import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import copy
import io
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
	'''
	The positional encoding class is used in the encoder and decoder layers.
	It's role is to inject sequence order information into the data since self-attention
	mechanisms are permuatation equivariant. Naturally, this is not required in the static
	transformer since there is no concept of 'order' in a portfolio.'''

	def __init__(self,window,d_model):
		super().__init__()

		pe = torch.zeros(window, d_model)
		for pos in range(window):
			for i in range(0, d_model, 2):
			  pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
				
			for i in range(1, d_model, 2):
			  pe[pos, i] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))             
				
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		seq_len = x.size(1)
		x = x + Variable(self.pe[:,:seq_len],requires_grad=False)
		return x

def create_mask(x,y):
	'''Create masks to be used in the decoder.
	First has dimension [1,prediction_window,prediction_window]
	Second has dimension [1,prediction_window,context_window]
	'''
	nopeak_mask = np.triu(np.ones((1,x+1,y+1)),k=1).astype('uint8')
	peak = torch.from_numpy(nopeak_mask)
	nopeak = peak[:,:-1,1:]
	peak_mask = nopeak.transpose(1,2)
	peak_mask = Variable(peak_mask)
	return peak_mask

def get_clones(module, N):
	'''
	This helper function is used to create copies of encoder and decoder layers.
	These copies of encoder/decoder layers are used to construct the
	complete stacked encoder/decoder modules.
	'''
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadAttention(nn.Module):
	'''This is a Mult-Head wide self-attention class.'''
	def __init__(self, heads, d_model, context_window, pred_window,
		first = None, dropout = 0.1, mask = None):
		super().__init__()
		
		self.h = heads
		self.dropout = nn.Dropout(dropout)
		self.mask = mask
		self.d_model = d_model
		self.pred_window = pred_window
		self.context_window = context_window
		self.first = first
		
	   
		self.q_linear = nn.Linear(d_model, heads*d_model,bias=False)
		self.v_linear = nn.Linear(d_model, heads*d_model,bias=False)
		self.k_linear = nn.Linear(d_model, heads*d_model,bias=False)
		  
		self.unifyheads = nn.Linear(heads*d_model, d_model)

	def forward(self, q, k, v):
		
		b,_,_ = q.size()
		
		if self.mask is not None:

			k = self.k_linear(k).view(b, self.pred_window, self.h, self.d_model)
			q = self.q_linear(q).view(b, self.pred_window, self.h, self.d_model)
			v = self.v_linear(v).view(b, self.pred_window, self.h, self.d_model)

			# Here we 'fold' the heads into the batch size to do the scaled dot-product computation.
			k = k.transpose(1, 2).contiguous().view(b*self.h, self.pred_window, self.d_model)
			q = q.transpose(1, 2).contiguous().view(b*self.h, self.pred_window, self.d_model)
			v = v.transpose(1, 2).contiguous().view(b*self.h, self.pred_window, self.d_model)

			scores = torch.matmul(q, k.transpose(1, 2))
			scores = scores/math.sqrt(self.d_model)

			# Use mask of size [1,pred,pred]
			masking = self.mask
			scores = scores.masked_fill(masking == 0,-1e9)

			# Create the attention matrix using row-wise softmax operation.
			scores = F.softmax(scores,dim=2)

			# Scaled dot-product.
			scores = torch.matmul(scores, v).view(b, self.h, self.pred_window, self.d_model)
			concat = scores.transpose(1, 2).contiguous().view(b, self.pred_window, self.h*self.d_model)

			# Linear mapping to reduce dimension -1 from heads*d_model to d_model.
			output = self.unifyheads(concat)

			# Scores if first == True is a tensor with size [batch*heads,prediction_window,prediction_window]
			# ScoresXvalues is [batch*heads,prediction_window,prediction_window]x[batch*heads,self.pred_window,self.d_model_dynamic]
			# Giving a final matrix of size [batch,prediction_window,d_model_dynamic] (after the unification of the heads)

			# Scores if first == False is a tensor with size [batch*heads,prediction_window,context_window]
			# ScoresXvalues is [batch*heads,prediction_window,context_window]x[batch*heads,context_window,d_model_dynamic]
			# Hence we end up with a final tensor of size:
			# [batch,prediction_window,d_model_static] (after the unification of the heads).

			# Need to change the mask for each of these cases!
			   
		else:

			if self.first == True:
		  
				k = self.k_linear(k).view(b, self.context_window, self.h, self.d_model)
				q = self.q_linear(q).view(b, self.context_window, self.h, self.d_model)
				v = self.v_linear(v).view(b, self.context_window, self.h, self.d_model)

				# Here we 'fold' the heads into the batch size to do the scaled dot-product computation.
				k = k.transpose(1, 2).contiguous().view(b*self.h, self.context_window, self.d_model)
				q = q.transpose(1, 2).contiguous().view(b*self.h, self.context_window, self.d_model)
				v = v.transpose(1, 2).contiguous().view(b*self.h, self.context_window, self.d_model)

				scores = torch.matmul(q, k.transpose(1, 2))
				scores = scores/math.sqrt(self.d_model)

				# Create the attention matrix using row-wise softmax operation.
				scores = F.softmax(scores,dim=2)

				# Scaled dot-product.
				scores = torch.matmul(scores, v).view(b, self.h, self.context_window, self.d_model)
				concat = scores.transpose(1, 2).contiguous().view(b, self.context_window, self.h*self.d_model)

				# Linear mapping to reduce dimension -1 from heads*d_model to d_model.
				output = self.unifyheads(concat)

			elif self.first == False:

				k = self.k_linear(k).view(b, self.context_window, self.h, self.d_model)
				q = self.q_linear(q).view(b, self.pred_window, self.h, self.d_model)
				v = self.v_linear(v).view(b, self.context_window, self.h, self.d_model)

				# Here we 'fold' the heads into the batch size to do the scaled dot-product computation.
				k = k.transpose(1, 2).contiguous().view(b*self.h, self.context_window, self.d_model)
				q = q.transpose(1, 2).contiguous().view(b*self.h, self.pred_window, self.d_model)
				v = v.transpose(1, 2).contiguous().view(b*self.h, self.context_window, self.d_model)

				scores = torch.matmul(q, k.transpose(1, 2))
				scores = scores/math.sqrt(self.d_model)

				# Use mask [1,p,c]
				# masking = self.mask
				# scores = scores.masked_fill(masking == 0,-1e9)

				# Create the attention matrix using row-wise softmax operation.
				scores = F.softmax(scores,dim=2)

				# Scaled dot-product.
				scores = torch.matmul(scores, v).view(b, self.h, self.pred_window, self.d_model)
				concat = scores.transpose(1, 2).contiguous().view(b, self.pred_window, self.h*self.d_model)

				# Linear mapping to reduce dimension -1 from heads*d_model to d_model.
				output = self.unifyheads(concat)
	
		return output

class FeedForward(nn.Module):
	'''This is a pointwise feedforward network.'''
	def __init__(self, d_model, dropout = 0.1):
		super().__init__()
		
		self.linear = nn.Sequential(
			nn.Linear(d_model,10 * d_model),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(10 * d_model,d_model)
		)
	
	def forward(self, x):
		x = self.linear(x)
		return x 

class DecoderLayer(nn.Module):
	'''Decoder Layer class'''
	def __init__(self, heads, d_model, context_window, pred_window, first_mask, dropout = 0.1):
		super().__init__()

		self.d_model = d_model
		self.heads = heads

		self.norm_1 = nn.LayerNorm(d_model)
		self.norm_2 = nn.LayerNorm(d_model)
		self.norm_3 = nn.LayerNorm(d_model)

		self.attn_1 = MultiHeadAttention(heads, d_model, context_window, pred_window,
			dropout = dropout, mask = first_mask)
		self.attn_2 = MultiHeadAttention(heads, d_model, context_window, pred_window, dropout = dropout,
			first = False)
		self.ff = FeedForward(d_model)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)

	def forward(self,x,enc_out):
	
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn_1(x2,x2,x2))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.attn_2(x2,enc_out,enc_out))
		x2 = self.norm_3(x)
		x = x + self.dropout_3(self.ff(x2))
		return x

class Decoder(nn.Module):
	'''Stacked Decoder layer.'''
	def __init__(self, N, pe_window, heads, d_model, context_window, pred_window, first_mask, dropout = 0.1):
		super().__init__()

		self.N = N
		self.pe = PositionalEncoding(pe_window, d_model)
		self.decoderlayers = get_clones(DecoderLayer(heads, d_model, context_window, pred_window, first_mask, dropout = dropout), N)
		self.norm = nn.LayerNorm(d_model)

	def forward(self,x,enc_out):
		x = self.pe(x)

		for i in range(self.N):
			x = self.decoderlayers[i](x,enc_out)
		return self.norm(x)

class EncoderLayer(nn.Module):
	'''Encoder layer class.'''
	def __init__(self, heads, d_model, context_window, pred_window, first_mask, dropout = 0.1):
		super().__init__()

		self.d_model = d_model
		self.heads = heads

		self.norm_1 = nn.LayerNorm(d_model)
		self.norm_2 = nn.LayerNorm(d_model)

		self.attn = MultiHeadAttention(heads, d_model, context_window, pred_window,
			dropout = dropout, mask = first_mask, first = True)
		self.ff = FeedForward(d_model)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
	
	def forward(self,x):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2,x2,x2))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.ff(x2))
		return x

class Encoder(nn.Module):
	'''Stacked encoder class.'''
	def __init__(self, N, pe_window, heads, d_model, context_window, pred_window, first_mask, dropout):
		super().__init__()

		self.N = N
		self.pe = PositionalEncoding(pe_window, d_model)
		self.dynamiclayers = get_clones(EncoderLayer(heads, d_model, context_window, pred_window, first_mask, dropout = dropout), N)
		self.norm = nn.LayerNorm(d_model)

	def forward(self,x):
		x = self.pe(x)

		for i in range(self.N):
		  x = self.dynamiclayers[i](x)
		return self.norm(x)

class Ml4fTransformer(nn.Module):
	'''
	Main transformer class
	Input to encoder has dimension: [batch,context_window,d_model_dynamic]
	The output has the same dimension. We then flatten the encoder output and do a
	linear mapping from the feature space to price space. Hence we have [batch,context_window].
	We then reshape this to get [batch,context_window,d_decode] dimension. The input to the
	decoder has dimension [batch,prediction_window,d_decode]. We then do the two-layer self-attention
	here which results in an output of size [batch,prediction_window,d_decode]. We reshape this to
	[batch,prediction_window] and then this is the final output of the model, which is first inverted, 
	and then passed on to the loss function.
	'''
	def __init__(self, experiment = 'return', d_model_e = 5, d_model_d = 1,
		N = 2, heads = 4, first_mask = create_mask(5, 5), dropout = 0.1,
		context_window = 15, pred_window = 5, pe_window = 15):
		super().__init__()
		
		self.d_model_d = d_model_d
		self.d_model_e = d_model_e
		self.context_window = context_window
		self.pred_window = pred_window
		self.dynamicencode = Encoder(N, pe_window, heads, d_model_e,
			context_window, pred_window, first_mask, dropout = dropout)

		self.learn = nn.Sequential(
			nn.Linear(context_window * d_model_e, context_window),
			nn.ReLU(),
			nn.Dropout(dropout)
			)
		
		self.decoder = Decoder(N, pe_window, d_model_d, heads, context_window, pred_window,
			first_mask, dropout = dropout)
		
		if experiment == 'return':

			self.map = nn.Sequential(
				nn.Linear(pred_window, pred_window),
				nn.ReLU(),
				nn.Dropout(dropout)
				)
	
		else:
	
			self.map = nn.Sequential(
				nn.Linear(pred_window, pred_window),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Sigmoid()
				)
	
	def forward(self,x,y):
		b = x.size(0)
		enc_output = self.dynamicencode(x)
		enc_output = enc_output.view(b,-1)
		enc_map = self.learn(enc_output)
		enc_map = enc_map.view(-1,self.window,self.d_model_d)
		dec_output = self.decoder(y,enc_map)
		dec_output = dec_output.view(b,self.pred_window)
		final = self.map(dec_output)
		return final
