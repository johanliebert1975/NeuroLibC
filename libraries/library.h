#pragma once


// HEADER FILES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//DEFINITIONS

    // OUTPUT STATES
#define SUCCESS 0;
#define ERROR -1;

    // DATA SIZE AND LENGTH
#define GRID_SIZE 64  // 8x8 grid flattened
#define MAX_LABEL_LEN 10

#define INPUT_NODES 64
#define HIDDEN_NODES_LAYER1 8
#define HIDDEN_NODES_LAYER2 8
#define OUTPUT_NODES 2

// DATA STRUCTURES
typedef struct {
    int grid[GRID_SIZE];  // Flattened 8x8 grid
    char label[MAX_LABEL_LEN];
} TrainingData;

typedef struct {
    int grid[GRID_SIZE];
}InputData;

// FUNCTION HEADERS

    // LOADING DATA FUNCTIONS
int load_training_data(const char *training_data, TrainingData **data, const char *log_file);
void load_weights(const char *filename, float weights[64][8], const char *log_file);
void load_biases(const char *filename, float biases[8], const char *log_file);

    // NETWORK FUNCTIONS
void forward_pass(float input[INPUT_NODES], float output[OUTPUT_NODES],float weights1[64][8], float weights2[8][8], float weights3[8][2], float biases1[8], float biases2[8], float biases3[2]);
void backward_pass(float output[OUTPUT_NODES], float input[INPUT_NODES], char *label, float weights1[64][8], float weights2[8][8], float weights3[8][2],float biases1[8], float biases2[8], float biases3[2]);
void update_weights_and_biases(float weights1[64][8], float weights2[8][8], float weights3[8][2],
                                float biases1[8], float biases2[8], float biases3[2],
                                float d_weights1[64][8], float d_weights2[8][8], float d_weights3[8][2],
                                float d_biases1[8], float d_biases2[8], float d_biases3[2],
                                float learning_rate);

    // FILE I/O FUNCTIONS
void save_weights(float weights[][8], int rows, int cols, const char* filename);
void save_biases(float biases[], int size, const char* filename);
void save_weights_and_biases();

    // TRAINING FUNCTION
void train(const char *training_data);
float compute_loss(float output[], char *label);
