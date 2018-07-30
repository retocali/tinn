#include "Tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// Data object.
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

// Learns and predicts hand written digits with 98% accuracy.
int main(int argc, char** argv) {

    // Set default values
    int nhid = 28;
    int dataLines = 1593;
    int nips = 256;
    int nops = 10;
    int iterations = 128;
    char dataPath[256] = "semeion.data";
    char nnetPath[256] = "saved.tinn";
    float anneal = 0.99f;
    float rate = 1.0f;
    bool loadExisting = false;
    bool trainExisting = false;

    char arg[256];
    char val[256];

    // Parse config file;
    FILE* file = fopen(argv[1], "r");
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
            memcpy(&dataPath, &val, 256);
        } else if(strcmp(arg, "NNET_PATH") == 0) {
            memcpy(&nnetPath, &val, 256);
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
        }
    }

    srand(time(0));
    // Load the training set
    Data data = build(dataPath, nips, nops, dataLines);
    Tinn tinn;
    if (loadExisting) {
        // Load neural network
        tinn = xtload(nnetPath);
    } else {
        // Build neural network
        tinn = xtbuild(nips, nhid, nops);
    }

    if (trainExisting || !loadExisting) {
        for(int i = 0; i < iterations; i++) {
            shuffle(data);
            float error = 0.0f;
            for(int j = 0; j < data.rows; j++) {
                const float* const in = data.in[j];
                const float* const tg = data.tg[j];
                error += xttrain(tinn, in, tg, rate);
            }
            printf("error %.12f :: learning rate %f\n",
                (double) error / data.rows,
                (double) rate);
            rate *= anneal;
        }
        xtsave(tinn, nnetPath);
    }
    
    shuffle(data);
    const float* const in = data.in[0];
    const float* const tg = data.tg[0];
    const float* const pd = xtpredict(tinn, in);

    // Prints target
    xtprint(tg, data.nops);
    // Prints prediction
    xtprint(pd, data.nops);

    // All done. Let's clean up.
    xtfree(tinn);
    dfree(data);
    return 0;
}
