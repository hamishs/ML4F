'''
Evaluation function.

- Position-wise analysis of sequence predictions for individual stocks.
- Plot predicted returns and prices against their corresponding true values for individual stocks.
- Loss plot for inference.

'''

class Performance():
  '''
  loss is the list of losses for each epoch.
  y is the true values.
  y_hat is the output of the best model during inference.
  batch is just batch size.
  '''
  def __init__(self,loss,y,y_hat,batch):
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
