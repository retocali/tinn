# Build & Run
    
Build the executable 
    
    make

Run the executable

    ./tinn <path_to_config>

# Training Data Info
The training data consists of hand written digits written both slowly and quickly.
Each line in the data set corresponds to one handwritten digit. Each digit is 16x16 pixels in size
giving 256 inputs to the neural network.

At the end of the line 10 digits signify the hand written digit:

    0: 1 0 0 0 0 0 0 0 0 0
    1: 0 1 0 0 0 0 0 0 0 0
    2: 0 0 1 0 0 0 0 0 0 0
    3: 0 0 0 1 0 0 0 0 0 0
    4: 0 0 0 0 1 0 0 0 0 0
    ...
    9: 0 0 0 0 0 0 0 0 0 1

This format can be used to train and test on any other data set.  

# Config File Terminology
- HIDDEN_LAYER_NODES - The number of nodes in the single hidden layer between the input and output
- DATA_LINES - The number of lines in the data file (excluding the last newline)
- NUM_INPUTS - The number of inputs in the neural net
- NUM_OUTPUTS - The number of outputs in the neural net
- TRAIN_ITERATIONS - The number of training iterations that will be run
- DATA_PATH - The path to the desired data file
- NNET_PATH - The path to the desired file containing neural net weights and biases
- ANNEAL - The change in learning rate applied after each training iteration
- LEARNING_RATE - The rate at which weights are adjusted 
- LOAD_EXISTING - YES=Load existing neural network  NO=Create new neural network
- TRAIN_EXISTING - YES=Train the network NO=Continue without training the network
- MANUAL_TESTING - YES=Test network manually NO=Automatically test all of training set
