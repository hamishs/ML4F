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

def accuracy_calc(x,y):
	'''
	This is a function used to calculate accuracy in the price movement experiment.
	If the absolute value of the difference between the output and the labels is < 0.5, 
	then the output is an accurate prediction
	'''
	s,t = x.size()
	total = s*t
	
	diff = torch.abs(x-y)
	one = torch.ones(s,t)
	zero = torch.zeros(s,t)

	accuracy = torch.where(diff<0.5,one,zero)
	acc_total = torch.sum(accuracy)
	
	percent = acc_total/total
	return percent


def loaders(train_data, val_data, test_data,
	batch_train = 56, batch_val = 142, batch_test = 426):
	'''Helper function to create DataLoaders for training.'''

	train_loader = DataLoader(train_data, batch_size=batch_train, shuffle=False)
	val_loader = DataLoader(val_data, batch_size=batch_val,shuffle=False)
	test_loader = DataLoader(test_data, batch_size=batch_test,shuffle=False)

	return train_loader, val_loader, test_loader

class Ml4fDataset(Dataset):
	'''Custom dataset class so that we can load input, future sequence
	and label data simultaneously during training.'''
	def __init__(self, input_data, label_data, token_data):
		
		self.input_data = input_data
		self.token_data = token_data
		self.label_data = label_data
		
	def __len__(self):
		
		return len(self.input_data)
	
	def __getitem__(self, index):
		
		x = self.input_data[index]
		y_tok = self.token_data[index]
		y = self.label_data[index]
		return x,y,y_tok

class DataPreProcess():
	'''This class handles all data pre-processing for the model.'''
	def __init__(self, url_input, url_label, potent, window = 8,
		label = 'return',
		pred_window = 3, d_model_e = 5):
	
		self.url_input = url_input
		self.url_label = url_label
		self.window = window
		self.pred_window = pred_window
		self.label = label
		self.d_model_e = d_model_e
		self.potent = potent
	
  
	def loading_input_data(self):
		''' Load input data.'''
	
		port_df = pd.read_csv(self.url_input)
		port = torch.tensor(port_df.to_numpy())
		return port

	def loading_label_data(self):
		''' Load label data. Setting self.label to False removes the return
		labels which are not required for price prediction.'''

		label_df = pd.read_csv(self.url_label)
		label = torch.tensor(label_df.to_numpy())
	 
		if self.label=='return':
		
			return_label = label[self.window:,self.potent:-self.potent].double()
			return_token = label[self.window-1:-self.pred_window,self.potent:-self.potent].double()
			return return_label, return_token
  
		elif self.label=='hit':
	  
			hit_label = label[self.window:,-self.potent:].double()
			hit_token = label[self.window-1:-self.pred_window,-self.potent:].double()
			return hit_label, hit_token
	
		elif self.label=='price':
	  
			price_label = label[self.window:,:self.potent].double()
			price_token = label[self.window-1:-self.pred_window,:self.potent].double()
			return prices_label, hit_token

	def shift_window(self):
		'''Creates the input sequences and their corresponding future sequences.'''
	
		x = self.loading_input_data()
		original_tens = x[0:self.window]
		L = x.size(0)
		total_window = self.window + self.pred_window - 1
	
		for i in range(1,L-total_window):
			train_input = x[i:i+self.window]
			original_tens = torch.cat((original_tens,train_input),dim=0)

		return original_tens


	def shift_pred(self):

		y, label_token = self.loading_label_data()
		label_tens = y[0:self.pred_window]
		L = y.size(0)

		for i in range(1,(L+1)-self.pred_window):
			train_label = y[i:i+self.pred_window]
			label_tens = torch.cat((label_tens,train_label),dim=0)

		return label_tens, label_token

	def portfolio_stack(self):

		x = self.shift_window()
		x = x.view(-1,self.window,self.potent*self.d_model_e)
		y,y_token = self.shift_pred()
		y = y.view(-1,self.pred_window*self.potent)

		x_train = x[:1400].view(-1,self.window,self.d_model_e)
		x_val = x[1420:1562].view(-1,self.window,self.d_model_e)
		x_test = x[1562:1988].view(-1,self.window,self.d_model_e)

		y_train = y[:1400].view(-1,self.pred_window)
		y_val = y[1420:1562].view(-1,self.pred_window)
		y_test = y[1562:1988].view(-1,self.pred_window)
	
		y_token_train = y_token[:1400].contiguous().view(-1,1)
		y_token_val = y_token[1420:1562].contiguous().view(-1,1)
		y_token_test = y_token[1562:1988].contiguous().view(-1,1)
	
		sizes_check = (x_train.size(), y_train.size(),y_token_train.size())

		train_data = Ml4fDataset(x_train,y_train,y_token_train)
		val_data = Ml4fDataset(x_val,y_val,y_token_val)
		test_data = Ml4fDataset(x_test,y_test,y_token_test)

		return train_data, val_data, test_data, sizes_check

	
class Normalisation():
	'''We normalise each feature of the input data using MinMaxScaling -
	there are 5. But we only want the max and min values for the close prices. 
	This is the 3rd index. Note that we do batch-wise normalisation since we
	believe this helps avoid the negative impact of outlier values
	which may occur from using longer time periods.'''

	def __init__(self, context_window = 8, predict_window = 3, d_model_e = 5, d_model_d = 1):
	
		self.d_model_e = d_model_e
		self.predict_window = predict_window
		self.context_window = context_window
		self.d_model_d = d_model_d

	def normal(self, x): 
		max_cols = []
		min_cols = []
		
		x = torch.log(x)
		x = x.view(-1, self.d_model_e)
		one,two = x.size()
		x_norm = torch.zeros((one,two),dtype=torch.double)
		for i in range(x.size(1)):
			x_col = x[:,i:i+1]
			max = torch.max(x_col,dim=0)[0]
			max_cols.append(max) 
			min = torch.min(x_col,dim=0)[0]
			min_cols.append(min)
			x_col = (x_col-min)/(max-min)
			x_norm[:,i:i+1] = x_col
		x_norm = x_norm.view(-1,self.context_window,self.d_model_e)

		price_max = torch.exp(max_cols[3])
		price_min = torch.exp(min_cols[3])

		self.prev_max = max_cols[3]
		self.prev_min = min_cols[3]

		return x_norm, price_max, price_min

	def normal_future(self, y, min_ = None, max_ = None):
		''' Log transform and min, max scaling of y. '''
		if min_ is None:
			min_ = self.prev_min
		if max_ is None:
			max_ = self.prev_max

		y = torch.log(y)
		y = (y - min_) / (max_ - min_)
		return y
	
