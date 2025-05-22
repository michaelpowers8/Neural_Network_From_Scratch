import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression,make_classification
import hashlib
import sys

def get_variable_info():
    # Get the current global and local variables
    globals_dict = globals()
    locals_dict = locals()
    
    # Combine them, prioritizing locals (to avoid duplicates)
    all_vars = {**globals_dict, **locals_dict}
    
    # Filter out modules, functions, and built-ins
    variable_info:list[dict[str,str|int|float|list|set|dict|bytes]] = []
    for name, value in all_vars.items():
        # Skip special variables, modules, and callables
        if name.startswith('__') and name.endswith('__'):
            continue
        if callable(value):
            continue
        if isinstance(value, type(sys)):  # Skip modules
            continue
            
        # Get variable details
        var_type:str = type(value).__name__
        try:
            var_hash:str = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
        except Exception:
            var_hash:str = "Unhashable"
        
        var_size:int = sys.getsizeof(value)
        
        variable_info.append({
            "Variable Name": name,
            "Type": var_type,
            "Hash": var_hash,
            "Size (bytes)": var_size
        })
    
    # Convert to a DataFrame for nice tabular output
    df:pd.DataFrame = pd.DataFrame(variable_info)
    return df

def initialize_parameters(input_size:int, hidden_layer_size:int, output_size:int):
    """
    Initialize random parameters for a new neural network.
    
    :param input_size: The number of features in the training set.
    :param hidden_layer_size: The number of neurons that will run calculations on all inputs (Can be any integer, typically a power of 2).
    :param outpus_size: The number of classes the neural network can choose from in a classification problem.
    :return: All the weights and biases that will be used to run the training process in gradient descent for the neural network.
    """
    W1 = np.random.rand(input_size,hidden_layer_size) * 0.01 # First weight that will be multiplied by all of the inputs
    b1 = np.random.rand(hidden_layer_size,1) # First constant bias term that will be added after each input is multiplied by W1
    W2 = np.random.rand(hidden_layer_size,output_size) * 0.01 # Second weight that will be applied after the activation function (typically ReLU function)
    b2 = np.random.rand(output_size,1) # Second constant bias term that will be added after each hiiden layer input is multiplied by W2

    return W1,b1,W2,b2

def ReLU(Z:np.ndarray) -> np.ndarray:
    """
    Apply the ReLU function which is a piecewise function which states ReLU(x)={x if x>0 else 0}

    :param Z: The resulting array after applying the dot product to W1 and the input array plus the b1 bias.
    :return: Numpy array where all values of the input are the same, but all values that were originally less than 0 are now set to 0.
    """
    return np.maximum(Z,0)

def derivative_ReLU(Z:np.ndarray) -> np.ndarray: 
    """
    Apply the derivative of the ReLU function on the array Z. For every element, i, in Z, if i>0, Zi=1, else Zi=0
    """
    return Z > 0

def softmax(Z:np.ndarray) -> np.ndarray:
    """
    For every element in the input Z of length n, element i, raise e^Zi and then divide that by the sum of all elements of e^Zn 

    :param Z: The resulting array after applying the dot product to W2 and the array resulting from applying the ReLU function.

    :return: Numpy array with all the probabilities of each possible outcome occurring. 

    Example
    --------
    >>> For 3 classes: resulting softmax -> [0.03  0.91  0.06] which means class at index 1 has the highest probability
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propogation(W1:np.ndarray, b1:np.ndarray, W2:np.ndarray, b2:np.ndarray, X:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Apply the forward propogation process to the neural network.

    :param W1: Array of weights that the dot product will be applied to on the input array of features.
    :param b1: Array of constants of bias that will be added after the dot product of W1 and X completes.
    :param W2: Array of weights that the dot product will be applied to on the hidden layer array.
    :param b2: Array of constants of bias that will be added after the dot product of W2 and the hidden layer complete.
    :param X: The input array of features of the data set.

    Return
    --------
    >>> Z1 -> The resulting array of the dot product of X and W1 plus b1
    >>> A1 -> Array after applying ReLU function to Z1
    >>> Z2 -> The resulting array of the dot product of A1 and W2 plus b2
    >>> A2 -> Array after applying ReLU function to Z2.
    """
    Z1:np.ndarray = W1.dot(X) + b1
    A1:np.ndarray = ReLU(Z1)
    Z2:np.ndarray = W2.dot(A1) + b2
    A2:np.ndarray = softmax(Z2) # Predictions of probabilities of classifications
    
    return Z1, A1, Z2, A2

def one_hot_encode(Y:np.ndarray) -> np.ndarray:
    """
    Set every classification to strictly consist of 0's and 1's for binary vectorization of classifications and remove categorical data.

    :param Y: Array of answers that the neural network is attempting to guess
    :return: Array with c rows and s columns where c is the number of classes in the classification and s is the size of the input Y
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Create a new matrix of 0's where there are as many rows as Y has rows, and the number of columns is equal to the number of different classes in Y assuming classes are structured from the integers 0 to n-1.
    one_hot_Y[np.arange(Y.size), Y] = 1 # Add one 1 to every row in this array in the specified class position, so if the 3rd element of Y is 5, and the size of Y is 100, then the 3rd element in row 6 of this array will be set to 1
    one_hot_Y = one_hot_Y.T # Transpose the array so that columns are examples instead of rows to be examples
    return one_hot_Y

def backwards_propogation(Z1:np.ndarray, A1:np.ndarray, Z2:np.ndarray, A2:np.ndarray, W1:np.ndarray, W2:np.ndarray, X:np.ndarray, Y:np.ndarray):
    m:int = Y.size
    one_hot_Y:np.ndarray = one_hot_encode(Y)
    dZ2:np.ndarray = A2 - one_hot_Y # Measures how much the output layer is off compared to the actual expected answer
    dW2 = (1 / m) * dZ2.dot(A1.T) # Derivative of the loss function with the respect to the weights in layer 2
    db2 = (1 / m) * np.sum(dZ2) # Average of the absolute error. On average, how far off was the model from the answer 
    dZ1:np.ndarray = W2.T.dot(dZ2) * derivative_ReLU(Z1) # Apply the weights of errors from layer 2 onto layer 1 multiplied by the derivative of the activation function applied to Z1
    dW1 = (1 / m) * (dZ1.dot(X.T))
    db1 = (1 / m) * np.sum(dZ1)

    return dW1, db1, dW2, db2

def main():    
    get_variable_info().to_json("Neural_Network_End_Variables.json",orient='table',indent=4)

if __name__ == "__main__":
    random_state:int = 42

    data = np.column_stack(make_classification(n_samples=10_000,n_features=10,n_informative=3,n_redundant=1,random_state=random_state,n_classes=3,n_clusters_per_class=2))
    training_data = data[0:(round(len(data)*0.8))]
    test_data = data[(round(len(data)*0.8)):]
    # data = np.array(data) # Convert DataFrame to ndarray to allow mathematical manipulation easier and more efficient

    m,n = training_data.shape # M x N matrix where M is currently the number of example rows and N is the number of features per set plus the target column
    np.random.shuffle(data) # Shuffle data before splitting data into train and validating sets that are used in training

    data_valid = training_data[:round(len(training_data)*0.1)].T # Cross validation set to ensure overfitting not occurring. Transposed so columns represent images
    y_valid = data_valid[-1] # Answer is the last value
    X_valid = data_valid[:n-1] # Features that make up the data starting at index 0 which is the first feature to n which is the number of pixels as found in data.shape

    data_train = training_data[round(len(training_data)*0.1):m].T # Actual training set that the neural network will see and learn from
    y_train = data_train[-1] # Answer is the last value
    X_train = data_train[:n-1] # Features that make up the data starting at index 0 which is the first feature to n which is the number of pixels as found in data.shape

    print(X_train.shape)
    print(y_train)

    main()