#include "library.h"

int Load_TrainingData(TrainingData* _TrainingData){
    FILE* TrInputfile = fopen("Training Data/TrainingDataInputs.csv", "r");
    FILE* TrOutputfile = fopen("Training Data/TrainingDataOutputs.csv", "r");

    if (TrOutputfile == NULL) {
        perror("Error opening Training Data Output file");
        fclose(TrInputfile);
        free(_TrainingData);  // Free allocated memory in case of failure
        return EXIT_FAILURE;
    }

    if (TrOutputfile == NULL) {
        perror("Error opening Training Data Output file");
        fclose(TrInputfile);
        free(_TrainingData);  // Free allocated memory in case of failure
        return EXIT_FAILURE;
    }
    if (ValidateFile(TrInputfile) == 0 && ValidateFile(TrOutputfile) == 0)
    {
        printf("Started Loading Training Data\n");
    }
    
    int inputbufferSize = 256;
    if(inputbufferSize < INPUT_LAYER_SIZE)
    {
        printf("Buffer size readjusted\n");
        inputbufferSize = 2048;
    }
    
    char inputbuffer[inputbufferSize];  // Increase buffer size to handle longer lines
    char outputbuffer[200]; // Increase buffer size to handle longer lines

    int data_index = 0;
    
    // Read input data
    while (fgets(inputbuffer, sizeof(inputbuffer), TrInputfile) != NULL && fgets(outputbuffer, sizeof(outputbuffer), TrOutputfile) != NULL) {
        char* token = strtok(inputbuffer, ",");
        _TrainingData[data_index].grid[0] = atof(token);
        for (size_t i = 1; i < INPUT_LAYER_SIZE; i++) {
            token = strtok(NULL, ",");
            _TrainingData[data_index].grid[i] = atof(token);
        }

        token = strtok(outputbuffer, ",");
        _TrainingData[data_index].label[0] = atof(token);
        token = strtok(NULL, ",");
        _TrainingData[data_index].label[1] = atof(token);
        
        data_index++;
    }

    fclose(TrInputfile);
    fclose(TrOutputfile);

    printf("Sucessfully Loaded the Training Data\n");
    return data_index;
}

int Save_Weights_Biases(Classifier* _Classifier){
    FILE* Weights = fopen("Weights\\Weights_Layer1.csv","w");
    if (Weights == NULL)
    {
        perror("Error Opening Weights_Layer1");
        return ERROR;
    }
    for (size_t i = 0; i < INPUT_LAYER_SIZE; i++)
    {
        for (size_t j = 0; j < HIDDEN_LAYER1_SIZE; j++)
        {
            fprintf(Weights,"%.2f,",_Classifier->Weights_Layer1[i][j]);
        }
        fprintf(Weights,"\n");
    }
    fclose(Weights);

    Weights = fopen("Weights\\Weights_Layer2.csv","w");
    if (Weights == NULL)
    {
        perror("Error Opening Weights_Layer2");
        return ERROR;
    }
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++)
    {
        for (size_t j = 0; j < HIDDEN_LAYER2_SIZE; j++)
        {
            fprintf(Weights,"%.2f,",_Classifier->Weights_Layer2[i][j]);
        }
        fprintf(Weights,"\n");
    }
    fclose(Weights);

    Weights = fopen("Weights\\Weights_Layer3.csv","w");
    if (Weights == NULL)
    {
        perror("Error Opening Weights_Layer3");
        return ERROR;
    }
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++)
    {
        for (size_t j = 0; j < OUTPUT_LAYER_SIZE; j++)
        {
            fprintf(Weights,"%.2f,",_Classifier->Weights_Layer3[i][j]);
        }
        fprintf(Weights,"\n");
    }
    fclose(Weights);

    FILE* Biases = fopen("Biases\\Biases_Layer1.csv","w");
    if (Biases == NULL)
    {
        perror("Error Opening Biases_Layer1.csv");
        return ERROR;
    }
    for (size_t i = 0; i < HIDDEN_LAYER1_SIZE; i++)
    {
        fprintf(Biases,"%.2f,",_Classifier->Biases_Layer1[i]);
    }
    fclose(Biases);

    Biases = fopen("Biases\\Biases_Layer2.csv","w");
    if (Biases == NULL)
    {
        perror("Error Opening Biases_Layer2.csv");
        return ERROR;
    }
    for (size_t i = 0; i < HIDDEN_LAYER2_SIZE; i++)
    {
        fprintf(Biases,"%.2f,",_Classifier->Biases_Layer2[i]);
    }
    fclose(Biases);
    
    Biases = fopen("Biases\\Biases_Layer3.csv","w");
    if (Biases == NULL)
    {
        perror("Error Opening Biases_Layer3.csv");
        return ERROR;
    }
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        fprintf(Biases,"%.2f,",_Classifier->Biases_Layer3[i]);
    }
    fclose(Biases);

    printf("Saved Weights and Biases to respective files\n");
    return SUCCESS;
}

int Load_Weights_Biases(Classifier* _Classifier) {
    FILE* Weights = fopen("Weights/Weights_Layer1.csv", "r");
    if (Weights == NULL) {
        perror("Error opening Weights_Layer1.csv");
        return ERROR;
    }

    char buffer[1024]; // Use a large enough buffer to hold a line
    size_t i = 0;
    while (fgets(buffer, sizeof(buffer), Weights) != NULL && i < INPUT_LAYER_SIZE) {
        char* token = strtok(buffer, ",");
        size_t j = 0;
        while (token != NULL && j < HIDDEN_LAYER1_SIZE) {
            _Classifier->Weights_Layer1[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(Weights); // Don't forget to close the file

    Weights = fopen("Weights/Weights_Layer2.csv","r");
    if (Weights == NULL)
    {
        perror("Error Opening Weights_Layer2.csv");
        return ERROR;
    }
    i = 0;
    while (fgets(buffer,sizeof(buffer),Weights) != NULL && i < HIDDEN_LAYER1_SIZE)
    {
        size_t j = 0;
        char* token = strtok(buffer,",");
        while (token != NULL && j < HIDDEN_LAYER2_SIZE)
        {
            _Classifier->Weights_Layer2[i][j] = atof(token);
            token = strtok(NULL,",");
            j++;
        }
        i++;
    }
    fclose(Weights);

    Weights = fopen("Weights/Weights_Layer3.csv","r");
    if (Weights == NULL)
    {
        perror("Error Opening Weights_Layer3.csv");
        return ERROR;
    }
    i = 0;
    while (fgets(buffer,sizeof(buffer),Weights) && i<HIDDEN_LAYER2_SIZE)
    {
        char* token = strtok(buffer,",");
        size_t j =0;
        while (token != NULL && j< OUTPUT_LAYER_SIZE)
        {
            _Classifier->Weights_Layer3[i][j] = atof(token);
            token = strtok(NULL,",");
            j++;
        }
        i++;
    }
    fclose(Weights);
    
    // Load Biases for Layer 1
    FILE* Biases = fopen("Biases/Biases_Layer1.csv", "r");
    if (Biases == NULL) {
        perror("Error Opening Biases_Layer1.csv");
        return ERROR;
    }

    if (fgets(buffer, sizeof(buffer), Biases) != NULL) {
        char* token = strtok(buffer, ",");
        for (size_t i = 0; i < HIDDEN_LAYER1_SIZE && token != NULL; i++) {
            _Classifier->Biases_Layer1[i] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(Biases);

    // Load Biases for Layer 2
    Biases = fopen("Biases/Biases_Layer2.csv", "r");
    if (Biases == NULL) {
        perror("Error Opening Biases_Layer2.csv");
        return ERROR;
    }

    if (fgets(buffer, sizeof(buffer), Biases) != NULL) {
        char* token = strtok(buffer, ",");
        for (size_t i = 0; i < HIDDEN_LAYER2_SIZE && token != NULL; i++) {
            _Classifier->Biases_Layer2[i] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(Biases);

    // Load Biases for Layer 3
    Biases = fopen("Biases/Biases_Layer3.csv", "r");
    if (Biases == NULL) {
        perror("Error Opening Biases_Layer3.csv");
        return ERROR;
    }

    if (fgets(buffer, sizeof(buffer), Biases) != NULL) {
        char* token = strtok(buffer, ",");
        for (size_t i = 0; i < OUTPUT_LAYER_SIZE && token != NULL; i++) {
            _Classifier->Biases_Layer3[i] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(Biases);
    
    printf("Successfully Loaded the Weights and biases into the Classifier\n");

    return SUCCESS;
}