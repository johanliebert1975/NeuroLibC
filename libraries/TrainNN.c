#include "library.h"

int Forward_Propagation(Classifier* _Classifier, float* InputData) {
   
    // Load the input data to the Classifier's InputData array
    for (size_t i = 0; i < INPUT_LAYER_SIZE; i++) {
        _Classifier->InputData[i] = InputData[i];  // Store the input in the structure
    }
    // FIRST LAYER PROPAGATION (Input to Hidden Layer 1)
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        _Classifier->HiddenLayer1Data[i] = 0;
        for (size_t j = 0; j < INPUT_LAYER_SIZE; j++) {
            _Classifier->HiddenLayer1Data[i] += _Classifier->Weights_Layer1[j][i] * _Classifier->InputData[j];
        }
        _Classifier->HiddenLayer1Data[i] += _Classifier->Biases_Layer1[i];
        _Classifier->HiddenLayer1Data[i] = relu(_Classifier->HiddenLayer1Data[i]);  // Apply ReLU
    }
    
    // SECOND LAYER PROPAGATION (Hidden Layer 1 to Hidden Layer 2)
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        _Classifier->HiddenLayer2Data[i] = 0;
        for (size_t j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            _Classifier->HiddenLayer2Data[i] += _Classifier->Weights_Layer2[j][i] * _Classifier->HiddenLayer1Data[j];
        }
        _Classifier->HiddenLayer2Data[i] += _Classifier->Biases_Layer2[i];
        _Classifier->HiddenLayer2Data[i] = relu(_Classifier->HiddenLayer2Data[i]);  // Apply ReLU
    }
    
    // THIRD LAYER PROPAGATION (Hidden Layer 2 to Output Layer)
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        _Classifier->OutputData[i] = 0;
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            _Classifier->OutputData[i] += _Classifier->Weights_Layer3[j][i] * _Classifier->HiddenLayer2Data[j];
        }
        _Classifier->OutputData[i] += _Classifier->Biases_Layer3[i];
        _Classifier->OutputData[i] = sigmoid(_Classifier->OutputData[i]);  // Apply Sigmoid
    }

    return SUCCESS;
}

int Back_Propagation(Classifier* _Classifier,float* Expected_Output){
    float OutputError[OUTPUT_LAYER_SIZE];
    float HiddenLayer2Error[HIDDEN_LAYER2_SIZE];
    float HiddenLayer1Error[HIDDEN_LAYER1_SIZE];

    float learning_rate = 0.008;

    // 1. Compute output layer error (for sigmoid + binary cross-entropy)
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        OutputError[i] = _Classifier->OutputData[i] - Expected_Output[i];
    }
    // 2. Backpropagate error to the second hidden layer
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        HiddenLayer2Error[i] = 0;
        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            HiddenLayer2Error[i] += OutputError[j] * _Classifier->Weights_Layer3[i][j];
        }
        HiddenLayer2Error[i] *= relu_derivative(_Classifier->HiddenLayer2Data[i]);
    }

    // 3. Backpropagate error to the first hidden layer
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        HiddenLayer1Error[i] = 0;
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            HiddenLayer1Error[i] += HiddenLayer2Error[j] * _Classifier->Weights_Layer2[i][j];
        }
        HiddenLayer1Error[i] *= relu_derivative(_Classifier->HiddenLayer1Data[i]);
    }

    // 4. Update weights and biases for output layer
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            _Classifier->Weights_Layer3[j][i] -= learning_rate * OutputError[i] * _Classifier->HiddenLayer2Data[j];
        }
        _Classifier->Biases_Layer3[i] -= learning_rate * OutputError[i];
    }

    // 5. Update weights and biases for second hidden layer
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        for (size_t j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            _Classifier->Weights_Layer2[j][i] -= learning_rate * HiddenLayer2Error[i] * _Classifier->HiddenLayer1Data[j];
        }
        _Classifier->Biases_Layer2[i] -= learning_rate * HiddenLayer2Error[i];
    }

    // 6. Update weights and biases for first hidden layer
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        for (size_t j = 0; j < INPUT_LAYER_SIZE; j++) {
            _Classifier->Weights_Layer1[j][i] -= learning_rate * HiddenLayer1Error[i] * _Classifier->InputData[j];
        }
        _Classifier->Biases_Layer1[i] -= learning_rate * HiddenLayer1Error[i];
    }

    return SUCCESS;
}

int Train_Classifier(Classifier* _Classifier, TrainingData* _TrainingData, int epochs){
    int num_Samples = Load_TrainingData(_TrainingData);
    Load_Weights_Biases(_Classifier);

    // Loop over epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_cost = 0.0;

        // Loop over samples in the training set
        for (size_t i = 0; i < num_Samples; i++) {
            // Forward propagation
            Forward_Propagation(_Classifier, _TrainingData[i].grid);

            // Compute the cost (assuming Binary Cross-Entropy)
            float cost = Binary_Cross_Entropy_Cost(_Classifier->OutputData, _TrainingData[i].label);
            total_cost += cost;

            // Backpropagation
            Back_Propagation(_Classifier, _TrainingData[i].label);
        }

        // Average cost for the epoch
        printf("Epoch %d: Average Cost = %.4f\n", epoch, (total_cost / num_Samples));
        
    return SUCCESS;
    }
}