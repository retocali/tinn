# Building

First you need to download the training data and update the DATA_PATH in tinn.config

    wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data
    
Build the executable 
    
    make

Run the executable

    ./tinn <path_to_config>

# Running

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

This gives 10 outputs to the neural network. The test program will output the
accuracy for each digit. Expect above 99% accuracy for the correct digit, and
less that 0.1% accuracy for the other digits.
