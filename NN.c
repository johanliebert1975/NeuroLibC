#include "library.h"


int main() {
    Classifier* _Classifier = (Classifier*)malloc(sizeof(Classifier));
    TrainingData* _TrainingData = (TrainingData*)malloc(100*sizeof(TrainingData));

//---------------------- INITIALIZE NEW RANDOM WEIGHTS AND BIASES ------------------------------------------
   /* NOTE when trying to initialize the weights and biases always use the srand function in the main */
   
    // Initialize_Weights_Biases(_Classifier);
    // Save_Weights_Biases(_Classifier);

//----------------------- TRAIN NEURAL NETWORK ---------------------------------------------------------------
    
    // float initial_learning_rate = 0.00001;

    // Load_Weights_Biases(_Classifier);
    // Load_TrainingData(_TrainingData);

    // Train_Classifier(_Classifier,_TrainingData,5,initial_learning_rate);
    // Save_Weights_Biases(_Classifier);

// ------------------- PREDICT THE DIGIT --------------------------------------------------------------------

    // Load_Weights_Biases(_Classifier);
    // float input[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0};
    
    // Forward_Propagation(_Classifier,input);
    // for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    // {
    //     printf("%.2f ",_Classifier->OutputData[i]);
    // }
    
    free(_Classifier);
    free(_TrainingData);

    return EXIT_SUCCESS;
}
