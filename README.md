# Neural Network From Scratch  

A Python implementation of a neural network built from scratch (without ML libraries like TensorFlow/PyTorch) for educational purposes. Includes forward/backpropagation, activation functions, and training on sample datasets.  

## Features  
- **Customizable Architecture**: Define layers, neurons, and activation functions (Sigmoid, ReLU).  
- **Training & Evaluation**: Implements gradient descent, loss calculation (MSE), and accuracy metrics.  
- **Modular Design**: Separates core components (e.g., `Layer`, `Network` classes).  
- **Dataset Handling**: Loads training data from JSON (`data.json`) for input/output pairs.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/michaelpowers8/Neural_Network_From_Scratch.git  
   cd Neural_Network_From_Scratch  
   ```  
2. Ensure Python 3.x and dependencies are installed:  
   ```bash  
   pip install numpy matplotlib  
   ```  

## Usage  
1. **Configure Network**: Modify `network_config.json` to set layers, learning rate, etc.:  
   ```json  
   {  
     "input_size": 2,  
     "hidden_layers": [4, 3],  
     "output_size": 1,  
     "learning_rate": 0.01,  
     "epochs": 1000  
   }  
   ```  
2. **Prepare Data**: Add input/output pairs to `data.json`.  
3. **Train the Network**: Run the main script:  
   ```bash  
   python neural_network.py  
   ```  

## Code Overview  
Key files:  
- `neural_network.py`: Core implementation (forward/backpropagation, training loop).  
- `activations.py`: Activation functions (Sigmoid, ReLU) and their derivatives.  
- `data.json`: Sample training data (e.g., XOR problem or linear regression).  

Example output:  
```  
Epoch 100/1000 | Loss: 0.1234  
Epoch 200/1000 | Loss: 0.0456  
...  
Test Accuracy: 92.3%  
```  

## Dependencies  
- Python 3.x  
- NumPy (`pip install numpy`)  
- Matplotlib (optional, for visualization)  

## License  
MIT License (see `LICENSE`).  