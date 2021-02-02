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

def split_data(inputs, label, future, window = 15,split_one = 1600, split_two = 1864):
	'''Helper function for splitting data into train, validation and test splits.'''
	input_train = inputs[:split_one,:]
	input_val = inputs[split_one+window:split_two,:]
	input_test = inputs[split_two:,:]

	future_train = future[:split_one,:]
	future_val = future[split_one+window:split_two,:]
	future_test = future[split_two:,:]

	lab_train = label[:split_one,:]
	lab_val = label[split_one+window:split_two,:]
	lab_test = label[split_two:,:]

	return input_train, input_val, input_test, future_train, future_val, future_test, lab_train, lab_val, lab_test

class Ml4fDataset(Dataset):
	'''Custom dataset class so that we can load input, future sequence
	and label data simultaneously during training.'''
	def __init__(self, input_data, label_data, future_data):
		self.input_data = input_data
		self.future_data = future_data
		self.label_data = label_data
		
	def __len__(self):
		return len(self.input_data)
	
	def __getitem__(self, index):
		x = self.input_data[index]
		x_fut = self.future_data[index]
		y = self.label_data[index]
		return x,x_fut,y

class DataPreProcess():
	'''This class handles all data pre-processing for the model.'''
	def __init__(self,url_input,url_label, window = 15,
		label = 'return', portfolio_size = 1,
		pred_window = 5, d_model_e = 5):
	
		self.url_input = url_input
		self.url_label = url_label
		self.window = window
		self.pred_window = pred_window
		self.label = label
		self.portfolio_size = portfolio_size
		self.d_model_e = d_model_e
  
	# Load input data.
	def loading_input_data(self):
	
		port_df = pd.read_csv(self.url_input)
		port = port_df.to_numpy()
		port = torch.Tensor(port)
		return port

	# Load label data. Setting label to False removes the return labels which are not required for price prediction.
	def loading_label_data(self):
	
		label_df = pd.read_csv(self.url_label)
		label_nump = label_df.to_numpy()

		if self.label=='return':
			label_drop = np.delete(label_nump, slice(0, self.window), 0)
			return_label = np.delete(label_drop,slice(0,73,3),1)
			return_label = torch.tensor(np.delete(return_label,slice(0,49,2),1))
			return_label = return_label.double()  
			return return_label
  
		elif self.label=='hit':
			hit_drop = np.delete(label_nump, slice(0, self.window), 0)
			hit_label = np.delete(hit_drop,slice(0,73,3),1)
			hit_label = torch.tensor(np.delete(hit_label,slice(1,50,2),1))
			hit_label = hit_label.double()
			return hit_label  
	
		elif self.label=='price':
			prices_drop = np.delete(label_nump, slice(0, self.window), 0)
			prices_label = np.delete(prices_drop,slice(1,74,3),1)
			prices_label = torch.Tensor(np.delete(prices_label,slice(1,50,2),1))
			prices_label = prices_label.double()
			return prices_label


	# This method creates the input sequences and their corresponding future sequences.
	def shift_window(self):
	
		x = self.loading_input_data()
		original_tens = x[0:self.window]
		L = x.size(0)
		total_window = self.window + self.pred_window - 1
	
		for i in range(1,L-total_window):
			train_input = x[i:i+self.window]
			original_tens = torch.cat((original_tens,train_input),dim=0)

		future_tens = x[self.window:]
		future_seq = future_tens[0:self.pred_window]
		L_future = future_tens.size(0)

		for i in range(1,(L_future+1)-self.pred_window):
			future_input = future_tens[i:i+self.pred_window]
			future_seq = torch.cat((future_seq,future_input),dim=0)

		return original_tens, future_seq


	# This method creates the price labels for each future sequence.
	def shift_pred(self):

		y = self.loading_label_data()
		label_tens = y[0:self.pred_window]
		L = y.size(0)

		for i in range(1,(L+1)-self.pred_window):
			train_label = y[i:i+self.pred_window]
			label_tens = torch.cat((label_tens,train_label),dim=0)

		return label_tens


  # The original datasets have dimension [2015,125] whereby each stock has a representation of dimension 5 (so 25 stocks). 
  # (Note the 0th dimension varies based on what we choose for the context and prediction window) 
  # Since we have moved to doing single stock prediction, we have to modify the dataset accordingly. We want to take the training 
  # subset of each stock and stack them into one single training set, and do the same for the validation and test sets. The splits are 1600 and 1900,
  # meaning for a single portfolio the training set now has size 1600*25 = 40,000. This method executes this modification for us and 
  # returns datasets using the custom dataset class.


  # Here we use the y labels as the future sequence.
	
	def portfolio_stack(self):
		x,x_fut = self.shift_window()
		y = self.shift_pred()

		potent = 25

		x_prime = x[:,:self.portfolio_size*self.d_model_e]
		x_prime = x_prime.view(-1,self.window,self.d_model_e)

		# Use y_prime for the labels and the future sequence values

		y_prime = y[:,:self.portfolio_size]
		y_prime = y_prime.contiguous().view(-1,self.pred_window*self.portfolio_size)
		x_tr, x_v, x_te, x_fut_tr, x_fut_v, x_fut_te, y_tr, y_v, y_te = split_data(x_prime,y_prime,y_prime)

		for pos in range(1,potent):
		
			x_new = x[:,pos*self.portfolio_size*self.d_model_e:(pos+1)*self.portfolio_size*self.d_model_e]
			x_new = x_new.view(-1,self.window,self.d_model_e)

			y_new = y[:,pos*self.portfolio_size:(pos+1)*self.portfolio_size]
			y_new = y_new.contiguous().view(-1,self.pred_window*self.portfolio_size)

			x_new_tr, x_new_v, x_new_te, x_fut_new_tr, x_fut_new_v, x_fut_new_te, y_new_tr, y_new_v, y_new_te = split_data(x_new,y_new,y_new)

			x_tr = torch.cat((x_tr,x_new_tr),dim=0)
			x_v = torch.cat((x_v,x_new_v),dim=0)
			x_te = torch.cat((x_te,x_new_te),dim=0)
			x_fut_tr = torch.cat((x_fut_tr, x_fut_new_tr),dim=0)
			x_fut_v = torch.cat((x_fut_v,x_fut_new_v),dim=0)
			x_fut_te = torch.cat((x_fut_te,x_fut_new_te),dim=0)
			y_tr = torch.cat((y_tr,y_new_tr),dim=0)
			y_v = torch.cat((y_v,y_new_v),dim=0)
			y_te = torch.cat((y_te,y_new_te),dim=0)

		sizes_check = (x_tr.size(), y_tr.size(), x_fut_tr.size())

		train_data = Ml4fDataset(x_tr,y_tr,x_fut_tr)
		val_data = Ml4fDataset(x_v,y_v,x_fut_v)
		test_data = Ml4fDataset(x_te,y_te,x_fut_te)

		return train_data, val_data, test_data, sizes_check

def loaders(train_data, val_data, test_data,
	batch_train = 20, batch_val = 66, batch_test = 66):
	'''Helper function to create DataLoaders for training.'''

	train_loader = DataLoader(train_data, batch_size=batch_train, shuffle=False)
	val_loader = DataLoader(val_data, batch_size=batch_val,shuffle=False)
	test_loader = DataLoader(test_data, batch_size=batch_test,shuffle=False)

	return train_loader, val_loader, test_loader

class Normalisation():
	'''We normalise each feature of the input data using MinMaxScaling -
	there are 5. But we only want the max and min values for the close prices. 
	This is the 3rd index. Note that we do batch-wise normalisation since we
	believe this helps avoid the negative impact of outlier values
	which may occur from using longer time periods.'''

	def __init__(self, context_window = 15, predict_window = 5, d_model_e = 5, d_model_d = 1):
	
		self.d_model_e = d_model_e
		self.predict_window = predict_window
		self.context_window = context_window
		self.d_model_d = d_model_d


	def normal(self,x): 
		max_cols = []
		min_cols = []
		
		x = torch.log(x)
		x = x.view(-1,self.d_model_e)
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

		price_max = max_cols[3]
		price_max = torch.exp(price_max)
		price_min = min_cols[3]
		price_min = torch.exp(price_min)
		return x_norm, price_max, price_min

	def normal_future(self,y):
		
		y = torch.log(y)
		y = y.view(-1,self.d_model_d)
		max = torch.max(y,dim=0)[0]
		min = torch.min(y,dim=0)[0]
		y = (y-min)/(max-min)
		y = y.view(-1, self.predict_window, self.d_model_d)
		return y

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
