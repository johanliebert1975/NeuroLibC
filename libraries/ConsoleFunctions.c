#include "library.h"

void DisplayArray(int* array, int size) {
    for (size_t i = 0; i < size; i++) {
        printf("%i ", array[i]);
    }
    printf("\n");
}

float x_rand(int min,int max){
     return ((float)(min + rand() % (max - min + 1))) / 100;
}

int Initialize_Weights_Biases(Classifier* _Classifier) {
    for (size_t i = 0; i < INPUT_LAYER_SIZE; i++) {
        for (size_t j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            _Classifier->Weights_Layer1[i][j] = x_rand(0, 400);
        }
    }
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            _Classifier->Weights_Layer2[i][j] = x_rand(0, 400);
        }
    }
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            _Classifier->Weights_Layer3[i][j] = x_rand(0, 400);
        }
    }
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        _Classifier->Biases_Layer1[i] = x_rand(-100, 100);
    }
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        _Classifier->Biases_Layer2[i] = x_rand(-100, 100);
    }
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        _Classifier->Biases_Layer3[i] = x_rand(-100, 100);
    }
    return SUCCESS;
}

//-------------------------------- ACTIVATION FUNCTIONS --------------------------------

// ReLU Activation Function
float relu(float x) {
    return x > 0 ? x : 0;
}

// ReLU Derivative (for backpropagation)
float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

// Sigmoid Activation Function
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// Sigmoid Derivative (for backpropagation)
float sigmoid_derivative(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

float Binary_Cross_Entropy_Cost(float* predicted_output, float* true_label) {
    float cost = 0.0;

    // Loop over each output (assuming a small number of output neurons, likely 1 or 2 for binary classification)
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        // Ensure predicted_output[i] is within valid range to avoid log(0)
        float predicted = fmax(fmin(predicted_output[i], 1.0 - 1e-7), 1e-7);
        
        cost += -true_label[i] * log(predicted) - (1.0 - true_label[i]) * log(1.0 - predicted);
    }

    // Return the average cost over the output neurons
    return cost / OUTPUT_LAYER_SIZE;
}
