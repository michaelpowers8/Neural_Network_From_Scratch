import numpy as np
import pandas as pd
import pickle
import hashlib
import sys
from typing import Any
import matplotlib.pyplot as plt

def get_variable_info(local_variables:dict[str,Any]):
    # Get the current global and local variables
    globals_dict = globals()
    
    # Combine them, prioritizing locals (to avoid duplicates)
    all_vars = {**globals_dict, **local_variables}
    
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

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def initialize_parameters(input_size:int, hidden_layer_size:int, num_layers:int, output_size:int) -> dict[str,np.ndarray]:
    """
    Initialize random parameters for a new neural network.
    
    :param input_size: The number of features in the training set.
    :param hidden_layer_size: The number of neurons that will run calculations on all inputs (Can be any integer, typically a power of 2).
    :param output_size: The number of classes the neural network can choose from in a classification problem.
    :return: All the weights and biases that will be used to run the training process in gradient descent for the neural network.
    """
    parameters:dict[str,np.ndarray] = {}
    parameters["W1"] = np.random.randn(hidden_layer_size,input_size) * np.sqrt(2 / input_size) # First weight that will be multiplied by all of the inputs
    parameters["b1"] = np.zeros((hidden_layer_size,1)) # First constant bias term that will be added after each input is multiplied by W1 (can also be an array of 0's)
    for layer in range(2,num_layers+1,1):
        parameters[f"W{layer:.0f}"] = np.random.randn(hidden_layer_size,hidden_layer_size) * np.sqrt(2 / hidden_layer_size) # Nth weight that will be applied after the activation function (typically ReLU function)
        parameters[f"b{layer:.0f}"] = np.zeros((hidden_layer_size,1)) # Nth constant bias term that will be added after each hiiden layer input is multiplied by W2(can also be an array of 0's)
    parameters[f"W{num_layers+1:.0f}"] = np.random.randn(output_size,hidden_layer_size) * np.sqrt(2 / hidden_layer_size) # Final weight that will be applied after the activation function (typically ReLU function)
    parameters[f"b{num_layers+1:.0f}"] = np.zeros((output_size,1)) # Final constant bias term that will be added after each hiiden layer input is multiplied by W2(can also be an array of 0's)

    return parameters

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

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    For every element in the input Z of length n, element i, raise e^Zi and then divide that by the sum of all elements of e^Zn 

    :param Z: The resulting array after applying the dot product to W2 and the array resulting from applying the ReLU function.

    :return: Numpy array with all the probabilities of each possible outcome occurring. 

    Example
    --------
    >>> For 3 classes: resulting softmax -> [0.03  0.91  0.06] which means class at index 1 has the highest probability
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propogation(parameters:dict[str,np.ndarray], X:np.ndarray) -> dict[str,np.ndarray]:
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
    propogation_results:dict[str,np.ndarray] = {}
    count:int = 1
    propogation_results[f"Z{1:.0f}"] = parameters["W1"].dot(X) + parameters["b1"]
    for _ in range((len(parameters)//2)-1):
        propogation_results[f"A{count:.0f}"] = swish(propogation_results[f"Z{count:.0f}"])
        count += 1
        propogation_results[f"Z{count:.0f}"] = parameters[f"W{count:.0f}"].dot(propogation_results[f"A{count-1:.0f}"]) + parameters[f"b{count:.0f}"]
    propogation_results[f"A{count:.0f}"] = softmax(propogation_results[f"Z{count:.0f}"])
    return propogation_results

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

def backwards_propogation(forward_propogation_results:dict[str,np.ndarray], parameters:dict[str,np.ndarray], X:np.ndarray, Y:np.ndarray) -> dict[str,np.ndarray]:
    """
    Perform the backwards propogation process to calculate the error in the gradient function and how much the weights and biases contributed to the errors.

    Original Code for exactly 2 layers:
    >>> m:int = Y.size
    >>> one_hot_Y:np.ndarray = one_hot_encode(Y) # See one_hot_encode_function above for details
    >>> dZ3:np.ndarray = A3 - one_hot_Y # Measures how much the output layer is off compared to the actual expected answer. More technically, the error gradient at the output layer
    >>> dW3 = (1 / m) * dZ3.dot(A2.T) # Derivative of the loss function with the respect to the weights in layer 3. Measures how much the weights contributed to the loss found in dZ3
    >>> db3 = (1 / m) * np.sum(dZ3) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ3
    >>> dZ2:np.ndarray = W3.T.dot(dZ3) * derivative_Swish(Z2) # Measures how much the output layer is off compared to the actual expected answer. More technically, the error gradient at the output layer
    >>> dW2 = (1 / m) * dZ2.dot(A1.T) # Derivative of the loss function with the respect to the weights in layer 2. Measures how much the weights contributed to the loss found in dZ2
    >>> db2 = (1 / m) * np.sum(dZ2) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ2
    >>> dZ1:np.ndarray = W2.T.dot(dZ2) * derivative_Swish(Z1) # Apply the weights of errors from layer 2 onto layer 1 multiplied by the derivative of the activation function applied to Z1
    >>> dW1 = (1 / m) * (dZ1.dot(X.T)) # Derivative of the loss function with the respect to the weights in layer 1. Measures how much the weights contributed to the loss found in dZ1
    >>> db1 = (1 / m) * np.sum(dZ1) # Average of the absolute error. On average, how far off was the model from the answer. Measures how much the biases contributed to the loss found in dZ1
    >>> return dW1, db1, dW2, db2, dW3, db3

    Return
    --------
    >>> dW1 -> The derivative of the loss function with respect to the weights in layer 1. Informs the neural network how much the input layer weights contributed to the loss.
    >>> db1 -> The mean average of the absolute error. Informs the neural network how much the bias terms contributed to the loss found at the hidden layer.
    >>> dW2 -> The derivative of the loss function with respect to the weights in layer 2. Informs the neural network how much the hidden layer weights contributed to the loss.
    >>> db2 -> The mean average of the absolute error. Informs the neural network how much the bias terms contributed to the loss found at the output layer.
    """
    m:int = Y.size
    one_hot_Y:np.ndarray = one_hot_encode(Y) # See one_hot_encode_function above for details
    backwards_propogation_results:dict[str,np.ndarray] = {}
    for key in range(len(forward_propogation_results)//2,0,-1):
        if(key==len(forward_propogation_results)//2):
            backwards_propogation_results[f"dZ{key:.0f}"] = forward_propogation_results[f"A{key:.0f}"] - one_hot_Y
        else:
            backwards_propogation_results[f"dZ{key:.0f}"] = parameters[f"W{key+1:.0f}"].T.dot(backwards_propogation_results[f"dZ{key+1:.0f}"]) * derivative_Swish(forward_propogation_results[f"Z{key:.0f}"])
        if(key==1):
            backwards_propogation_results[f"dW{key:.0f}"] = (1 / m) * (backwards_propogation_results[f"dZ{key:.0f}"].dot(X.T))
        else:
            backwards_propogation_results[f"dW{key:.0f}"] = (1 / m) * (backwards_propogation_results[f"dZ{key:.0f}"].dot(forward_propogation_results[f"A{key-1:.0f}"].T))
        backwards_propogation_results[f"db{key:.0f}"] = (1 / m) * np.sum(backwards_propogation_results[f"dZ{key:.0f}"])
    return backwards_propogation_results

def update_parameters(parameters:dict[str,np.ndarray],backwards_propogation_results:dict[str,np.ndarray],learning_rate:float):
    original_parameters:dict[str,np.ndarray] = parameters.copy()
    for key,item in original_parameters.items():
        parameters[key] = item - learning_rate*backwards_propogation_results[f"d{key}"]
    return parameters

def get_predictions(A_final:np.ndarray) -> np.ndarray:
    return np.argmax(A_final,0)

def get_accuracy(predictions:np.ndarray, Y:np.ndarray) -> float:
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X:np.ndarray, Y:np.ndarray, hidden_layer_size:int, number_of_layers:int, iterations:int, learning_rate:float) -> dict[str,np.ndarray]:
    unique_values = np.unique(Y, return_counts=False) # Finding all the unique class values in the array Y
    num_classes:int = len(unique_values)  # Count of unique values
    parameters:dict[str,np.ndarray] = initialize_parameters(input_size=X.shape[0],hidden_layer_size=32,num_layers=5,output_size=num_classes)
    for i in range(iterations):
        learning_rate = 0.01 * (0.95 ** (i // 100))  # Decay every 100 steps
        forward_propogation_results:dict[str,np.ndarray] = forward_propogation(parameters, X)
        backwards_propogation_results:dict[str,np.ndarray] = backwards_propogation(forward_propogation_results, parameters, X, Y)
        parameters = update_parameters(parameters, backwards_propogation_results, learning_rate)

        if(i % 100 == 0):
            print(f"Iterations complete: {i:,.0f}/{iterations:,.0f}")
            predictions = get_predictions(forward_propogation_results[next(reversed(forward_propogation_results))])
            print(f"Accuracy: {get_accuracy(predictions,Y)}\n")
    return parameters
        
def make_predictions(X:np.ndarray, parameters:dict[str,np.ndarray]):
    forward_propogation_results:dict[str,np.ndarray] = forward_propogation(parameters, X)
    predictions = get_predictions(forward_propogation_results[next(reversed(forward_propogation_results))])
    return predictions

def get_data() -> np.ndarray:
    train = pd.read_csv("mnist_train.csv",header=None)
    test = pd.read_csv("mnist_test.csv",header=None)
    data = pd.concat([train,test],axis=0)
    data = data.to_numpy()
    return data

def test_prediction(index:int, X:np.ndarray, Y:np.ndarray, parameters:dict[str,np.ndarray]):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], parameters)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def save_model(parameters:dict[str,np.ndarray]):
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(parameters, f)
    np.savez("trained_model.npz", **parameters)

def load_model():
    loaded = np.load("trained_model.npz")
    parameters = {key: loaded[key] for key in loaded.files}
    return parameters

def main():
    data:np.ndarray = get_data()
    np.random.shuffle(data)
    training_data = data[0:(round(len(data)*0.8))]
    test_data = data[(round(len(data)*0.8)):]

    m,n = training_data.shape # M x N matrix where M is currently the number of example rows and N is the number of features per set plus the target column
    np.random.shuffle(data) # Shuffle data before splitting data into train and validating sets that are used in training

    test_data = test_data.T # Cross validation set to ensure overfitting not occurring. Transposed so columns represent examples
    y_test:np.ndarray = test_data[0] # Answer is the first value
    y_test:np.ndarray = y_test.astype(int)
    X_test:np.ndarray = test_data[1:n] # Features that make up the data starting at index 0 which is the first feature to n which is the number of features as found in data.shape

    training_data = training_data.T # Actual training set that the neural network will see and learn from
    y_train:np.ndarray = training_data[0] # Answer is the first value
    y_train:np.ndarray = y_train.astype(int)
    X_train:np.ndarray = training_data[1:n] # Features that make up the data starting at index 0 which is the first feature to n which is the number of features as found in data.shape

    X_train = X_train / 255.0
    X_test = X_test / 255.0


    # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    try:
        parameters:dict[str,np.ndarray] = load_model()
        train_predictions = make_predictions(X_train, parameters)
        test_predictions = make_predictions(X_test, parameters)
        accuracy_train:float = get_accuracy(train_predictions,y_train)
        accuracy_test:float = get_accuracy(test_predictions,y_test)
    except:
        parameters:dict[str,np.ndarray] = gradient_descent(X_train, y_train, 32, 1, 1_000, 0.001)
        train_predictions = make_predictions(X_train, parameters)
        test_predictions = make_predictions(X_test, parameters)
        accuracy_train:float = get_accuracy(train_predictions,y_train)
        accuracy_test:float = get_accuracy(test_predictions,y_test)

    print(f"NN Prediction accuracy on training set: {accuracy_train}\nNN Prediction accuracy on test set: {accuracy_test}")

    for index in range(len(X_test)):
        test_prediction(index,X_test,y_test,parameters)

    save_model(parameters)
    get_variable_info(locals()).to_json("Neural_Network_End_Variables.json",orient='table',indent=4)

if __name__ == "__main__":
    main()