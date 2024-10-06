#include "library.h"

float* Load_Unclassified_Data(float* input){
    FILE* input_file = fopen("UnclassifiedInput.csv","r");
    if(ValidateFile(input_file)){
        return NULL;
    }
    char buffer[2048];
    while (fgets(buffer,sizeof(buffer),input_file) != NULL)
    {
        char* token = strtok(buffer,",");
        for (size_t i = 0; i < INPUT_LAYER_SIZE; i++)
        {
            input[i] = atof(token);
            token = strtok(NULL,",");
        }
    }
    return input;
}

int main() {
    srand(time(0));
    Classifier* _Classifier = (Classifier*)malloc(sizeof(Classifier));
    TrainingData* _TrainingData = (TrainingData*)malloc(100*sizeof(TrainingData));

//---------------------- INITIALIZE NEW RANDOM WEIGHTS AND BIASES ------------------------------------------

   /* NOTE when trying to initialize the weights and biases always use the srand function in the main */

    // Initialize_Weights_Biases(_Classifier);
    // Save_Weights_Biases(_Classifier);

//----------------------- TRAIN NEURAL NETWORK ---------------------------------------------------------------
    
    // float initial_learning_rate = 0.0001;

    // Train_Classifier(_Classifier,_TrainingData,50,initial_learning_rate);
    // Save_Weights_Biases(_Classifier);

// ------------------- PREDICT THE DIGIT --------------------------------------------------------------------

    Load_Weights_Biases(_Classifier);
    float* input = (float*)malloc(sizeof(float)*INPUT_LAYER_SIZE);
    Load_Unclassified_Data(input);
    
    Forward_Propagation(_Classifier,input);
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        printf("%.2f ",_Classifier->OutputData[i]);
    }
    printf("\n");
    if (_Classifier->OutputData[0]>_Classifier->OutputData[1])
    {
        printf("The Drawn Integer is a 2\n");
    }
    else{
        printf("The Drawn Integer is not a 2\n");
    }
    
    free(_Classifier);
    free(_TrainingData);

    return EXIT_SUCCESS;
}
