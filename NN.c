#include "library.h"


int main() {
    Classifier* _Classifier = (Classifier*)malloc(sizeof(Classifier));
    TrainingData* _TrainingData = (TrainingData*)malloc(100*sizeof(TrainingData));
    
    Train_Classifier(_Classifier,_TrainingData,3);

    printf("Weights Layer1\n");
    for (size_t i = 0; i < INPUT_LAYER_SIZE; i++)
    {
        for (size_t j = 0; j < HIDDEN_LAYER1_SIZE; j++)
        {
            printf("%.2f ",_Classifier->Weights_Layer1[i][j]);
        }
        printf("\n");
    }
    printf("Weights Layer 2\n");
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++)
    {
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++)
        {
            printf("%.2f ",_Classifier->Weights_Layer2[i][j]);
        }
        printf("\n");
    }
    printf("Weights Layer 3\n");\
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++)
    {
        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
        {
            printf("%.2f ",_Classifier->Weights_Layer2[i][j]);
        }
        printf("\n");
    }
    printf("New Output for the same Training Data: ");
    
    Forward_Propagation(_Classifier,_TrainingData[0].grid);
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        printf("%.2f ",_Classifier->OutputData[i]);
    }
    

    free(_Classifier);
    free(_TrainingData);

    return EXIT_SUCCESS;
}
