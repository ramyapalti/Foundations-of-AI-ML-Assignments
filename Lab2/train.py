# ### Lab-2 Backpropogation

######  Feed Forward Networks and Backpropagation
## We will implement a fully connected Feed Forward Neural Network(FFNN) and train on datasets

## For the rest of the assignment, we say FFNN, short for Feed Forward Neural Network

## We implement FFNN consisiting of 1 input layer, 1 hidden layer and 1 output layer
## Activation function for the output layer is Sigmoid function
## We train the FFNN forward and backward pass functions using backpropagation
## Loss function to be used is Mean Squared Error(MSE)

## Modify your code only between the lines marked with "## TODO" and "## END TODO"

# ### Libraries
### Importing the libraries
import numpy as np
import pickle as pkl
import argparse
import warnings
from matplotlib import pyplot as plt
  
# suppress warnings
warnings.filterwarnings('ignore')

# ### Preprocessing
def preprocessing(X):
  """
  Implement Normalization for input image features i.e. norm of each image feature vector should be 1
  arguments - X : numpy array of shape (n_samples, 784)
  returns - X_out: numpy array of shape (n_samples, 784) after normalization
  """
  # apply normalization to input features
  min_X = np.min(X)
  max_X = np.max(X)
  dev = max_X - min_X + 1e-7
  #scaling 
  X_out = (X - min_X)/dev

  assert X_out.shape == X.shape

  return X_out


# ### Flatten the Input
class FlattenLayer:
    ## This class converts a multi-dimensional into 1-d vector
    def __init__(self, input_shape):
        ## arguments- input_shape : Original shape, tuple of ints
        self.input_shape = input_shape

    def forward(self, input):
        '''
        Converts a multi-dimensional into 1-d vector
        arguments-
          input : training data, numpy array of shape (n_samples , self.input_shape)

        returns:
          input: training data, numpy array of shape (n_samples , -1) -- 1-d vector
        '''
        # returns flattened input
        return input.reshape(len(input), -1)
        
    
    def backward(self, output_error, learning_rate):
        '''
        Converts back the passed array to original dimension 
        arguments-
        output_error :  numpy array 
        learning_rate: float

        returns:
        output_error: A reshaped numpy array to allow backward pass
        '''
        shape_new = (output_error.shape[0],) + self.input_shape
        #returns reshaped array
        return output_error.reshape(shape_new)

# ### Fully Connected Layer
class FCLayer:
    ## Implements a fully connected layer  
    
    def __init__(self, input_size, output_size):
        '''
        arguments:
         input_size : Input shape, int
         output_size: Output shape, int 
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1., 1., (input_size, output_size)) # initilaise weights for this layer
        self.bias = np.zeros(self.output_size, dtype='float64')  # initilaise bias for this layer

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        arguments:
          input : training data, numpy array of shape (n_samples , self.input_size)

        returns:
           numpy array of shape (n_samples , self.output_size)
        '''
        
        # Implement forward pass of a fully connected network

        #save input here
        self.inp = input
        output = None

        ## TODO
        #  Use y = Xw + b to write output with input, weights and bias
        #  X is input, w is weights and b is bias
        
        ## END TODO
        
        # Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return output
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        arguments:
          output_error :  numpy array 
          learning_rate: float

        returns:
          Numpy array resulting from the backward pass
        '''
        #update weights here
        self.weights -= learning_rate*np.matmul(np.transpose(self.inp), output_error, dtype=np.float64)
        self.bias -= learning_rate*np.sum(output_error, axis = 0)
        
        #returns numpy array resulting from backward pass
        return np.matmul(output_error, np.transpose(self.weights), dtype=np.float64)
        

# ### Sigmoid Activation Layer
class SigmoidLayer:
    ## Implements a sigmoid layer which applies sigmoid function on the inputs. 
    ## Sigmoid function is used as the activation function for the output layer

    def __init__(self, input_size):
        self.input_size = input_size
        self.input = None
    
    def forward(self, input):
        '''
        Applies the sigmoid function 
        arguments:
          input : numpy array on which sigmoid function is to be applied

        returns:
           numpy array output from the sigmoid function
        '''

        output = []
        self.input = np.array(input, dtype=np.float64)
        ## TODO
        # implement sigmoid function
        
        ## END TODO
        return output
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Sigmoid layer
        arguments:
          output_error :  numpy array 
          learning_rate: float

        returns:
          Numpy array resulting from the backward pass
        '''
        val = 1/(1 + np.exp(-self.input))
        return (1 - val)*(val)*output_error

# ### Loss function
def mse(y_true, y_pred):
    # We implement loss function and gradient of loss function(for backpropogation)
    '''
    Implement Mean Sqaured Error(MSE) loss
    arguments:
        y_true :  Ground truth labels, numpy array 
        y_true :  Predicted labels, numpy array 
    returns:
       loss : float
    '''
    return np.mean(np.sum(np.square((y_true - y_pred)), axis = 1))/2.0


def mse_prime(y_true, y_pred):
    '''
    Implements derivative of MSE function, for the backward pass
    arguments:
        x :  numpy array 
    returns:
        Numpy array after applying derivative of MSE function
    '''
    return np.mean(y_pred - y_true, axis = 0)

def plot_graph(train_loss, test_loss):
    '''
    Plot the graph of train and test loss
    arguments:
        train_loss :  list of train loss
        test_loss :  list of test loss
    '''
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ### Training FFNN
def train(X_train, Y_train, step, hidden=12):

    '''
    Create and trains a feedforward network

    Do not forget to save the final weights of the feed forward network to a file. 
    Use these weights in the `predict` function 
    Args:
        X_train -- (num_test, 28, 28) for mnist
        Y_train -- (num_test, 28, 28) for mnist
    
    '''
     
    #define your network
    network = [
        FlattenLayer(input_shape=(28, 28)),
        FCLayer(28 * 28, hidden),
        SigmoidLayer(hidden),
        FCLayer(hidden, 10),
        SigmoidLayer(10),
    ] # This creates feed forward network 
    # input layer and 1 hidden layer connected with FC layer
    # Activation at hidden layer
    # hidden layer and output layer connected with FC Layer
    # 1 sigmoid layer for Activation

    # Choose appropriate learning rate and no. of epoch
    epochs = 40
    learning_rate = 0.01

    # train error
    tr_err = np.zeros(40, dtype=np.float64)
    # test error
    te_err = np.zeros(40, dtype=np.float64)

    with open(f"./data/mnist_test.pkl", "rb") as file:
        test_data = pkl.load(file)

    for epoch in range(epochs):
        n_samples = step
        error = 0
        
        for b in range(n_samples): # for each sample
            # x - current input, y - current output
            y = Y_train[b: b + 1]
            x = X_train[b: b + 1]

            output = x
            for l in network:
                # forward pass over all the layers here
                output = l.forward(output)

            # error to print
            err1 = np.zeros(output.shape, dtype=np.float64)
            err1[0][y] = 1
            
            #error calculation here
            error += mse(err1, output)
             
            #gradient calculation here
            output_error = mse_prime(err1, output)
             
            #performing back propogation
            for l in reversed(network):
                output_error = l.backward(output_error, learning_rate)

        ## printing error after each epoch
        error /= n_samples
        tr_err[epoch] = error # training error for current epoch, for plotting
        print('Epoch:%d/%d, error=%.4f' % (epoch + 1, epochs, error))

        #calculating testing error for current epoch
        sample = 900 # number of test samples
        error_sum_test = 0

        # for each test sample
        for i in range(sample):
            x_test = test_data[0][i].reshape(784, 1).T
            output = x_test
    
            for l in network:
                output = l.forward(output)

            err_temp = np.zeros(output.shape, dtype=np.float64) 
            err_temp[0][y] = 1
            
            # get error for current test sample
            error_sum_test += round(mse(output, err_temp),2)

        te_err[epoch] = error_sum_test/sample # scaling the error

    #Saving model weights here
    file = open(f"./data/mnist_weights.pkl", "wb")
    pkl.dump(network, file)

    #plot the training vs testing error graph 
    plot_graph(tr_err, te_err)


# ### Predictions
def predict(X_test, Y_true):
  """
  Predict the labels for the test data
  X_test -- np array of shape (num_test, 28, 28) for mnist.

  1. Load your trained weights from ./models/mnist_weights.pkl
  2. Initialize your model with your trained weights
  3. Compute the predicted labels and return it

  returns:
  Y_test - nparray of shape (num_test,)
  """

  n = pkl.load(open(f"./data/mnist_weights.pkl", 'rb'))   
  output = X_test
    
  for l in n:
    output = l.forward(output)
    
  # predicted labels
  Y_test = np.argmax(output, axis = 1)

  err1 = np.zeros(output.shape, dtype=np.float64)
  err1[0][Y_true] = 1
  assert Y_test.shape == (X_test.shape[0],) and type(Y_test) == type(X_test), "Check shape and type of Y_test"
  return [Y_test[0], round(mse(output, err1),2)]

parser = argparse.ArgumentParser(prog = 'train', description = 'FFNN')
parser.add_argument('--hidden', dest = 'hidden', type = str, default = '12', help = 'number of hidden layer nuerons')
                 

def main():
    args = parser.parse_args()
    hidden = args.hidden

    with open(f"./data/mnist_train.pkl", "rb") as file:
        train_mnist = pkl.load(file)
        num_train = 2000
        print(f"Input shape -- (2000, 28, 28), Output shape -- (2000, 1)")

    print("Training...")
    train(train_mnist[0],train_mnist[1], num_train, int(hidden))

if __name__ == "__main__":
    main()



