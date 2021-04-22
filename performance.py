import torch
import numpy as np 
import matplotlib.pyplot as plt
from pytorch_lightning import metrics

class Performance():
	'''
	Evaluation function.

	- Position-wise analysis of sequence predictions for individual stocks.
	- Plot predicted returns and prices against their corresponding true values for individual stocks.
	- Loss plot for inference.

	'''
	def __init__(self,loss,y,y_hat,batch):
		'''
		loss is the list of losses for each epoch.
		y is the true values.
		y_hat is the output of the best model during inference.
		batch is just batch size.
		'''
		self.y = y
		self.y_hat = y_hat
		self.positions = y.size(-1) 
		self.loss = loss
		self.batch = batch

	# returns two lists, one containing mse loss in each position and the other mae loss in each position for the best epoch.
	def metric(self,stock):
		absolute_error = []
		mse_error = []
		for i in range(self.positions):
			L_one = nn.L1Loss()
			L_two = nn.MSELoss()
			mae_loss = L_one(self.y[stock*self.batch:(stock+1)*self.batch,i],self.y_hat[stock*self.batch:(stock+1)*self.batch,i])
			absolute_error.append(mae_loss)
			mse_loss = L_two(self.y[stock*self.batch:(stock+1)*self.batch,i],self.y_hat[stock*self.batch:(stock+1)*self.batch,i])
			mse_error.append(mse_loss)
		return absolute_error, mse_error

	# This method plots the predictions vs the real values for whichever position you want.
	def series_plot(self,stock,target,experiment):
		y_t = self.y[stock*self.batch:(stock+1)*self.batch,target].numpy()
		y_hat_t = self.y_hat[stock*self.batch:(stock+1)*self.batch,target].detach().numpy()
		plt.figure()
		plt.plot(y_t)
		plt.plot(y_hat_t)
		plt.title('Predictions vs True')
		plt.xlabel('{}'.format(experiment))
		plt.ylabel('Days')

	# This method plots loss.
	def loss_plots(self):
		plt.figure()
		plt.plot(self.loss)
		plt.title('Loss')
		plt.xlabel('Epochs')
		plt.ylabel('MSE Loss')

class MetricManager:
	''' Class to manage calculations of performance metrics during training.
	Uses the API for pytorch_lightning.metrics'''
	
	def __init__(self, metric_base_list, names, device = 'cpu'):
		'''
		metric base list : list of classes of metrics from pytorch_lightning.metrics.
		names : name of each metric for printing
		'''
		self.metrics = [met().to(device) for met in metric_base_list]
		self.names = names
	
	def __call__(self, y_pred, y_tar):
		''' updates each metric with predictions y_pred and target y_tar'''
		for met in self.metrics:
			met.update(y_pred.detach(), y_tar)
	
	def reset(self):
		''' resets all metrics e.g. at start of epoch'''
		for met in self.metrics:
			met.reset()
	
	def compute(self):
		''' returns a list with current val of each metric'''
		out = [met.compute() for met in self.metrics]
		return out
	
	def print_out(self, start = '', end = ''):
		'''
		Prints a string with current val of each metric.
		start : optional starting string
		end : optional ending string
		'''
		if len(start) > 0:
			out = start + ' '
		else:
			out = ''

		for met, name in zip(self.metrics, self.names):
			res = met.compute().cpu().numpy()
			out += '{} {:.8f} '.format(name, res)
		out += end
		return out

class BinaryCrossEntropyMetric(metrics.Metric):
	'''Binary cross entropy implemented as a pytorch_lightning metric'''
	
	def __init__(self, **kwargs):
		super(BinaryCrossEntropyMetric, self).__init__(**kwargs)
		self.add_state('N', default = torch.tensor(0))
		self.add_state('loss', default = torch.tensor(0))
		self.BCE = torch.nn.BCELoss()
	
	def update(self, y_pred, y_tar):
		''' update the stored metric'''
		y_pred = y_pred.detach()
		y_tar = y_tar.detach()
		
		n_items = torch.numel(y_tar)
		batch = self.BCE(y_pred, y_tar)
		self.loss = self.loss * (self.N / (self.N + n_items)) + batch * (n_items / (self.N + n_items))
		self.N += n_items
	
	def compute(self):
		''' computes the current value of the metric'''
		return self.loss

class SignAccuracy(metrics.Metric):
	''' Accuracy on the signs of the label and prediction.'''

	def __init__(self, **kwargs):
		super(SignAccuracy, self).__init__(**kwargs)
		self.add_state('correct', default = torch.tensor(0), dist_reduce_fx = 'sum')
		self.add_state('total', default = torch.tensor(0), dist_reduce_fx = 'sum')

	def update(self, y_pred, y_tar):
		'''update the stored accuracy'''
		y_pred = (y_pred > 0)
		y_tar = (y_tar > 0).long()

		self.correct += torch.sum(y_pred == y_tar)
		self.total += y_tar.numel()

	def compute(self):
		''' computes the current value of the metric'''
		return self.correct.float() / self.total







