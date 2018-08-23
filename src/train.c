#include "Tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <wchar.h>
#include <locale.h>
#include <signal.h>

typedef struct {
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    int nips;
    // Number of outputs to neural network.
    int nops;
    // Number of rows in file (number of sets for neural network).
    int rows;
}
Data;

typedef struct {
    // Hidden layer nodes
    int nhid;
    // Training iterations
    int iterations;
    // Path to training data file
    char* dataPath;
    // Path to testing data file
    char* testPath;
    // Path to neural network file
    char* nnetPath;
    // Number of lines for training data
    int dataLines;
    // Number of lines for testing data
    int testLines;
    // Number of nodes in input layer
    int nips;
    // Number of nodes in output layer
    int nops;
    // Anneal of the learning learning rate
    float anneal;
    // Learning rate of the network
    float rate;
    // Should we load an existing network?
    bool loadExisting;
    // Should we train this network or run as-is?
    bool trainExisting;
    // Should we manually test this network?
    bool manualTesting;
}
Config;

// Reads a line from a file.
static char* readln(FILE* const file, int* size, char* line) {
    int ch = EOF;
    int reads = 0;

    while((ch = getc(file)) != '\n' && ch != EOF) {
        line[reads++] = ch;
        if(reads + 1 == *size)
            line = (char*) realloc((line), (*size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d(const int rows, const int cols) {
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata(const int nips, const int nops, const int rows) {
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char* line, const int row) {
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++) {
        const float val = 1.0f*(line[col]-'0');
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

// Frees a data object from the heap.
static void dfree(const Data d) {
    for(int row = 0; row < d.rows; row++) {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

static void cfgfree(const Config c) {
    free(c.dataPath);
    free(c.nnetPath);
    free(c.testPath);
}

// Randomly shuffles a data object.
static void shuffle(const Data d) {
    for(int a = 0; a < d.rows; a++) {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build(const char* path, const int nips, const int nops, int rows) {
    FILE* file = fopen(path, "r");
    if(file == NULL) {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        printf("Check file path in config file\n");
        exit(1);
    }
    Data data = ndata(nips, nops, rows);
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    for(int row = 0; row < rows; row++) {
        line = readln(file, &size, line);
        parse(data, line, row);
    }
    free(line);
    fclose(file);
    return data;
}

// Automatically tests network over full testing set
int beginAutoTesting(Data data, Tinn nnet, int lines) {
    printf( "\n####################################\n"
              "       BEGIN AUTO TESTING\n"
              "####################################\n\n");
    int correct = 0;
    for(int i=0;i<lines;i++){
        int target = xtgetLargestIndex(data.tg[i],10);
        float* prediction = xtpredict(nnet, data.in[i]);
        int result = xtgetLargestIndex(prediction,10);
        if(target == result) {
            printf("Test input was %d, the machine chose %d\n",target,result);
            correct++;
        } else {
            printf("Test input was %d, the machine chose %d - WRONG\n",target,result);
        }
    }
    float percent = (float)correct/(float)lines;
    printf("\nThe guessed with an accuracy of %.3f\n\n", percent);
    return correct;
}

// Tests the network based off user input
void beginManualTesting(Data data, Tinn nnet) {
    printf( "\n####################################\n"
              "       BEGIN MANUAL TESTING\n"
              "####################################\n\n");
    printf("Press any number key followed by enter or 'q' to quit: ");
    int ch = 0;
    while ((ch = getchar()) != 'q' && ch != 'Q') {
        shuffle(data);
        int target = -1;
        int index = -1;
        switch(ch) {
            case '0':
                while(target != 0) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '1':
                while(target != 1) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '2':
                while(target != 2) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '3':
                while(target != 3) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '4':
                while(target != 4) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '5':
                while(target != 5) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '6':
                while(target != 6) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '7':
                while(target != 7) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '8':
                while(target != 8) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
            case '9':
                while(target != 9) {target = xtgetLargestIndex(data.tg[++index],10);}
                break;
        }
        if(48<=ch && ch<=57) {
            printf("\n");
            xtprintImg(data.in[index], 16);
            float* prediction = xtpredict(nnet, data.in[index]);
            printf("You input %d but the machine chose:\n", target);
            xtprint(prediction, data.nops);
            printf( "=============================================================\n"
                    "=============================================================\n\n");
            printf("Press any number key followed by enter or 'q' to quit: ");
        }
    }
}

Config parseConfig(int argc, char** argv) {
    int nhid;
    int dataLines;
    int testLines;
    int nips;
    int nops;
    int iterations;
    char* dataPath = malloc(512*sizeof(char));
    char* nnetPath = malloc(512*sizeof(char));
    char* testPath = malloc(512*sizeof(char));
    float anneal;
    float rate;
    bool loadExisting;
    bool trainExisting;
    bool manualTesting;

    if(argc != 2) {
        printf("ERROR: Please add path to config file ./tinn <path>\n");
        exit(0);
    } else {
        printf("Attempting to read config file: %s\n", argv[1]);
    }
    FILE* file = fopen(argv[1], "r");
    if(file == NULL) {
        printf("ERROR: Bad File Path\n");
        exit(0);
    }
    char arg[512];
    char val[512];
    while(fscanf(file, "%s = %s\n", &arg, &val) != -1) {
        if(strcmp(arg, "HIDDEN_LAYER_NODES") == 0) {
            nhid = atoi(val);
        } else if(strcmp(arg, "DATA_LINES") == 0) {
            dataLines = atoi(val);
        } else if(strcmp(arg, "NUM_INPUTS") == 0) {
            nips = atoi(val);
        } else if(strcmp(arg, "NUM_OUTPUTS") == 0) {
            nops = atoi(val);
        } else if(strcmp(arg, "TRAIN_ITERATIONS") == 0) {
            iterations = atoi(val);
        } else if(strcmp(arg, "DATA_PATH") == 0) {
            memcpy(dataPath, &val, 512);
        } else if(strcmp(arg, "NNET_PATH") == 0) {
            memcpy(nnetPath, &val, 512);
        } else if(strcmp(arg, "ANNEAL") == 0) {
            anneal = atof(val);
        } else if(strcmp(arg, "LEARNING_RATE") == 0) {
            rate = atof(val);
        } else if(strcmp(arg, "LOAD_EXISTING") == 0) {
            if (strcmp(val, "YES") == 0) {
                loadExisting = true;
            } else {
                loadExisting = false;
            }
        } else if(strcmp(arg, "TRAIN_EXISTING") == 0) {
            if (strcmp(val, "YES") == 0) {
                trainExisting = true;
            } else {
                trainExisting = false;
            }
        } else if(strcmp(arg, "TEST_PATH") == 0) {
            memcpy(testPath, &val, 512);
        } else if(strcmp(arg, "TEST_LINES") == 0) {
            testLines = atoi(val);
        } else if(strcmp(arg, "MANUAL_TESTING") == 0) {
            if (strcmp(val, "YES") == 0) {
                manualTesting = true;
            } else {
                manualTesting = false;
            }
        }
    }
    const Config config = {
        nhid, iterations, dataPath, testPath, nnetPath, dataLines, testLines,
        nips, nops, anneal, rate, loadExisting, trainExisting, manualTesting
    };
    return config;
}

// Learns and predicts hand written digits with a high level of accuracy.
int main(int argc, char** argv) {
    printf( "\n####################################\n"
              "    THE VERY TINY NEURAL NETWORK\n"
              "####################################\n\n");

    // Parse config file
    Config c = parseConfig(argc, argv);

    // Set seed for shuffling data
    srand(2);

    // Build the training set
    Data data;
    if(c.trainExisting || !c.loadExisting) {
        printf("Building training set\n");
        data = build(c.dataPath, c.nips, c.nops, c.dataLines);
    } else {
        data = build(c.dataPath, c.nips, c.nops, 0);
    }

    // Build the testing set
    printf("Building testing set\n");
    Data testData = build(c.testPath, c.nips, c.nops, c.testLines);

    // Load/build neural network
    Tinn tinn;
    if (c.loadExisting) {
        // Load neural network
        printf("Loading existing neural network at %s\n", c.nnetPath);
        tinn = xtload(c.nnetPath);
    } else {
        // Build neural network
        printf("Building new neural network\n");
        tinn = xtbuild(c.nips, c.nhid, c.nops);
    }

    // Train network
    if (c.trainExisting || !c.loadExisting) {
        printf("Training neural network for %d iterations\n", c.iterations);
	int j = 0;
        for(int i = 0; i < c.iterations; i++) {
            shuffle(data);
            float error = 0.0f;
            for(int j = 0; j < data.rows; j++) {
                const float* const in = data.in[j];
                const float* const tg = data.tg[j];
                error += xttrain(tinn, in, tg, c.rate);
            }
            printf("iteration %d :: error %.12f :: learning rate %f\n",
                j,
                (double) error / data.rows,
                (double) c.rate);
            c.rate *= c.anneal;
	    j++;
        }
        printf("Saving neural network at %s\n", c.nnetPath);
        xtsave(tinn, c.nnetPath);
    }

    // Begin testing
    if(c.manualTesting) {
        beginManualTesting(testData, tinn);
    } else {
        beginAutoTesting(testData, tinn, c.testLines);
    }

    // Clean up and quit
    printf("Quitting program...\n\n");
    xtfree(tinn);
    dfree(data);
    dfree(testData);
    cfgfree(c);
    return 0;
}
