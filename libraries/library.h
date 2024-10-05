#pragma once

//---------------------------------------- HEADER FILES -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

//----------------------------------------- DEFINITIONS --------------------------------------------------------------

#define INPUT_LAYER_SIZE 64
#define OUTPUT_LAYER_SIZE 2

#define HIDDEN_LAYER1_SIZE 16
#define HIDDEN_LAYER2_SIZE 8

#define SUCCESS 0
#define ERROR -1

//---------------------------------------- DATA STRUCTURES -----------------------------------------------------------

typedef struct TrainingData {
    float grid[INPUT_LAYER_SIZE];
    float label[OUTPUT_LAYER_SIZE];
} TrainingData;

typedef struct NeuralNetwork{

    float InputData[INPUT_LAYER_SIZE];
    float HiddenLayer1Data[HIDDEN_LAYER1_SIZE];
    float HiddenLayer2Data[HIDDEN_LAYER2_SIZE];
    float OutputData[OUTPUT_LAYER_SIZE];

    float Weights_Layer1[INPUT_LAYER_SIZE][HIDDEN_LAYER1_SIZE];
    float Weights_Layer2[HIDDEN_LAYER1_SIZE][HIDDEN_LAYER2_SIZE];
    float Weights_Layer3[HIDDEN_LAYER2_SIZE][OUTPUT_LAYER_SIZE];

    float Biases_Layer1[HIDDEN_LAYER1_SIZE];
    float Biases_Layer2[HIDDEN_LAYER2_SIZE];
    float Biases_Layer3[OUTPUT_LAYER_SIZE];
    
}Classifier;

//--------------------------------------- FUNCTION HEADERS ----------------------------------------------------------
   
    // CONSOLE FUNCTIONS
void DisplayArray(int* array, int size);
float x_rand(int min,int max);
int Initialize_Weights_Biases(Classifier* _Classifier);

        // ACTIVATION FUNCTIONS
float relu(float x);
float relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);

        // COST FUNTION
float Binary_Cross_Entropy_Cost(float* predicted_output, float* true_label);

    // FILE_IO FUNCTIONS
int Load_TrainingData(TrainingData* _TrainingData);
int Save_Weights_Biases(Classifier* _Classifier);
int Load_Weights_Biases(Classifier* _Classifier);

    // TRAIN_NN FUNCTIONS
int Forward_Propagation(Classifier* _Classifier, float* InputData);
int Back_Propagation(Classifier* _Classifier,float* Expected_Output);
int Train_Classifier(Classifier* _Classifier, TrainingData* _TrainingData, int epochs);
