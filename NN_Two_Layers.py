import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
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
    :param output_size: The number of classes the neural network can choose from in a classification problem.
    :return: All the weights and biases that will be used to run the training process in gradient descent for the neural network.
    """
    W1 = np.random.rand(hidden_layer_size,input_size) * np.sqrt(2 / input_size) # First weight that will be multiplied by all of the inputs
    b1 = np.zeros((hidden_layer_size,1)) # First constant bias term that will be added after each input is multiplied by W1 (can also be an array of 0's)
    W2 = np.random.rand(hidden_layer_size,hidden_layer_size) * np.sqrt(2 / hidden_layer_size) # Second weight that will be applied after the activation function (typically ReLU function)
    b2 = np.zeros((hidden_layer_size,1)) # Second constant bias term that will be added after each hidden layer input is multiplied by W2(can also be an array of 0's)
    W3 = np.random.rand(output_size,hidden_layer_size) * np.sqrt(2 / hidden_layer_size) # Second weight that will be applied after the activation function (typically ReLU function)
    b3 = np.zeros((output_size,1)) # Second constant bias term that will be added after each hidden layer input is multiplied by W2(can also be an array of 0's)

    return W1,b1,W2,b2,W3,b3

def ReLU(Z:np.ndarray) -> np.ndarray:
    """
    Apply the ReLU function which is a piecewise function which states ReLU(x)={x if x>0 else 0}

    :param Z: The resulting array after applying the dot product to W1 and the input array plus the b1 bias.
    :return: Numpy array where all values of the input are the same, but all values that were originally less than 0 are now set to 0.
    """
    return np.maximum(Z,0)

def sigmoid(Z:np.ndarray) -> np.ndarray:
    return np.divide(1,1+np.exp(-Z))

def swish(Z:np.ndarray) -> np.ndarray:
    """
    Apply the Swish function on Z which is Swish(Z) = Z/(1+e^(-Z))

    :param Z: The resulting array after applying the dot product to W1 and the input array plus the b1 bias.
    :return: Numpy array where all values had the Swish function applied to them.
    """
    return Z*sigmoid(Z)

def derivative_ReLU(Z:np.ndarray) -> np.ndarray: 
    """
    Apply the derivative of the ReLU function on the array Z. For every element, i, in Z, if i>0, Zi=1, else Zi=0
    """
    return Z > 0

def derivative_Swish(Z:np.ndarray) -> np.ndarray:
    """
    If Sigmoid(Z) = 1/(1+e^-Z), and Swish(Z) = Z*(Sigmoid(Z)) = Z/(1+e^-Z), then d/dx(Sigmoid(Z)) = Swish(Z) - Z*(Sigmoid(Z)^2) + Sigmoid(Z)
    """
    sigmoid_Z:np.ndarray = sigmoid(Z)
    swish_Z:np.ndarray = swish(Z)
    return swish_Z-Z*np.power(sigmoid_Z,2)+sigmoid_Z

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

def forward_propogation(W1:np.ndarray, b1:np.ndarray, W2:np.ndarray, b2:np.ndarray, W3:np.ndarray, b3:np.ndarray, X:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
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
    >>> A2 -> Array after applying softmax function to Z2.
    """
    Z1:np.ndarray = W1.dot(X) + b1
    A1:np.ndarray = swish(Z1)
    Z2:np.ndarray = W2.dot(A1) + b2
    A2:np.ndarray = swish(Z2) 
    Z3:np.ndarray = W3.dot(A2) + b3
    A3:np.ndarray = softmax(Z3) # Predictions of probabilities of classifications
    
    return Z1, A1, Z2, A2, Z3, A3

def one_hot_encode(Y:np.ndarray) -> np.ndarray:
    """
    Set every classification to strictly consist of 0's and 1's for binary vectorization of classifications and remove categorical data.

    :param Y: Array of answers that the neural network is attempting to guess
    :return: Array with c rows and s columns where c is the number of classes in the classification and s is the size of the input Y
    """
    one_hot_Y = np.zeros((Y.size, int(Y.max() + 1))) # Create a new matrix of 0's where there are as many rows as Y has rows, and the number of columns is equal to the number of different classes in Y assuming classes are structured from the integers 0 to n-1.
    one_hot_Y[np.arange(Y.size), Y] = 1 # Add one 1 to every row in this array in the specified class position, so if the 3rd element of Y is 5, and the size of Y is 100, then the 3rd element in row 6 of this array will be set to 1
    one_hot_Y = one_hot_Y.T # Transpose the array so that columns are examples instead of rows to be examples
    return one_hot_Y

def backwards_propogation(Z1:np.ndarray, A1:np.ndarray, Z2:np.ndarray, A2:np.ndarray, Z3:np.ndarray, A3:np.ndarray, W1:np.ndarray, W2:np.ndarray, W3:np.ndarray, X:np.ndarray, Y:np.ndarray):
    """
    Perform the backwards propogation process to calculate the error in the gradient function and how much the weights and biases contributed to the errors.

    Return
    --------
    >>> dW1 -> The derivative of the loss function with respect to the weights in layer 1. Informs the neural network how much the input layer weights contributed to the loss.
    >>> db1 -> The mean average of the absolute error. Informs the neural network how much the bias terms contributed to the loss found at the hidden layer.
    >>> dW2 -> The derivative of the loss function with respect to the weights in layer 2. Informs the neural network how much the hidden layer weights contributed to the loss.
    >>> db2 -> The mean average of the absolute error. Informs the neural network how much the bias terms contributed to the loss found at the output layer.
    """
    m:int = Y.size
    one_hot_Y:np.ndarray = one_hot_encode(Y) # See one_hot_encode_function above for details
    dZ3:np.ndarray = A3 - one_hot_Y # Measures how much the output layer is off compared to the actual expected answer. More technically, the error gradient at the output layer
    dW3 = (1 / m) * dZ3.dot(A2.T) # Derivative of the loss function with the respect to the weights in layer 3. Measures how much the weights contributed to the loss found in dZ3
    db3 = (1 / m) * np.sum(dZ3) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ3
    dZ2:np.ndarray = W3.T.dot(dZ3) * derivative_Swish(Z2) # Measures how much the output layer is off compared to the actual expected answer. More technically, the error gradient at the output layer
    dW2 = (1 / m) * dZ2.dot(A1.T) # Derivative of the loss function with the respect to the weights in layer 2. Measures how much the weights contributed to the loss found in dZ2
    db2 = (1 / m) * np.sum(dZ2) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ2
    dZ1:np.ndarray = W2.T.dot(dZ2) * derivative_Swish(Z1) # Apply the weights of errors from layer 2 onto layer 1 multiplied by the derivative of the activation function applied to Z1
    dW1 = (1 / m) * (dZ1.dot(X.T)) # Derivative of the loss function with the respect to the weights in layer 1. Measures how much the weights contributed to the loss found in dZ1
    db1 = (1 / m) * np.sum(dZ1) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ1
 
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1:np.ndarray,b1:np.ndarray,W2:np.ndarray,b2:np.ndarray,W3:np.ndarray,b3:np.ndarray,dW1:np.ndarray,db1:np.ndarray,dW2:np.ndarray,db2:np.ndarray,dW3:np.ndarray,db3:np.ndarray,learning_rate:float):
    W1:np.ndarray = W1 - learning_rate*dW1
    b1:np.ndarray = b1 - learning_rate*db1
    W2:np.ndarray = W2 - learning_rate*dW2
    b2:np.ndarray = b2 - learning_rate*db2
    W3:np.ndarray = W3 - learning_rate*dW3
    b3:np.ndarray = b3 - learning_rate*db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3:np.ndarray) -> np.ndarray:
    return np.argmax(A3,0)

def get_accuracy(predictions:np.ndarray, Y:np.ndarray) -> float:
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X:np.ndarray, Y:np.ndarray, iterations:int, learning_rate:float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    unique_values = np.unique(Y, return_counts=False) # Finding all the unique class values in the array Y
    num_classes:int = len(unique_values)  # Count of unique values
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size=X.shape[0],hidden_layer_size=64,output_size=num_classes)
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propogation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backwards_propogation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)

        if(i % 100 == 0):
            print(f"Iterations complete: {i:,.0f}/{iterations:,.0f}")
            predictions = get_predictions(A3)
            print(f"Accuracy: {get_accuracy(predictions,Y)}\n")
    return W1, b1, W2, b2, W3, b3
        
def make_predictions(X:np.ndarray, W1:np.ndarray, b1:np.ndarray, W2:np.ndarray, b2:np.ndarray, W3:np.ndarray, b3:np.ndarray):
    _, _, _, _, _, A3 = forward_propogation(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def main():    
    random_state:int = 42
    np.random.seed(random_state)

    data = np.column_stack(make_classification(n_samples=10_000,n_features=8,n_informative=4,random_state=random_state,n_classes=2,n_clusters_per_class=2))
    training_data = data[0:(round(len(data)*0.8))]
    test_data = data[(round(len(data)*0.8)):]
    # data = np.array(data) # Convert DataFrame to ndarray to allow mathematical manipulation easier and more efficient

    m,n = training_data.shape # M x N matrix where M is currently the number of example rows and N is the number of features per set plus the target column
    np.random.shuffle(data) # Shuffle data before splitting data into train and validating sets that are used in training

    test_data = test_data.T # Cross validation set to ensure overfitting not occurring. Transposed so columns represent examples
    y_test:np.ndarray = test_data[-1] # Answer is the last value
    y_test:np.ndarray = y_test.astype(int)
    X_test:np.ndarray = test_data[0:n-1] # Features that make up the data starting at index 0 which is the first feature to n which is the number of features as found in data.shape

    training_data = training_data.T # Actual training set that the neural network will see and learn from
    y_train:np.ndarray = training_data[-1] # Answer is the last value
    y_train:np.ndarray = y_train.astype(int)
    X_train:np.ndarray = training_data[0:n-1] # Features that make up the data starting at index 0 which is the first feature to n which is the number of features as found in data.shape

    # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 10_000, 0.01)
    train_predictions = make_predictions(X_train, W1, b1, W2, b2, W3, b3)
    test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)
    accuracy_train:float = get_accuracy(train_predictions,y_train)
    accuracy_test:float = get_accuracy(test_predictions,y_test)

    print(f"NN Prediction accuracy on training set: {accuracy_train}\nNN Prediction accuracy on test set: {accuracy_test}")
    
    get_variable_info().to_json("Neural_Network_End_Variables.json",orient='table',indent=4)

if __name__ == "__main__":
    main()