#include "library.h"

float Weights_Layer1[64][8]; // Layer 1 weights
float Biases_Layer1[8];      // Layer 1 biases
float Weights_Layer2[8][8];  // Layer 2 weights
float Biases_Layer2[8];      // Layer 2 biases
float Weights_Layer3[8][2];  // Layer 3 weights
float Biases_Layer3[2];      // Layer 3 biases

float d_weights1[64][8] = {0};
float d_weights2[8][8] = {0};
float d_weights3[8][2] = {0};
float d_biases1[8] = {0};
float d_biases2[8] = {0};
float d_biases3[2] = {0};

float hidden1[HIDDEN_NODES_LAYER1] = {0};
float hidden2[HIDDEN_NODES_LAYER2] = {0};

float output[2] = {0};

float learning_rate = 0.01f;

char* log_filetrainingdata = "./Logfiles/logfiletrainingdata.txt";
char* log_fileloadweights = "./Logfiles/logfileloadweights.txt";
char* log_fileloadbiases = "./Logfiles/logfileloadbiases.txt";
// ACTIVATION FUNCTION

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
float relu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid_derivative(float output) {
    return sigmoid(output) * (1.0f - sigmoid(output));
}

float relu_derivative(float x) {
    return (x > 0) ? 1.0 : 0.0;
}


// FUNCTIONS TO TRAIN THE NETWORK

void log_message(const char *log_file, const char *message) {
    FILE *log = fopen(log_file, "a");
    if (log) {
        fprintf(log, "%s\n", message);
        fclose(log);
    }
}

void log_sampletrainingdata(const char *log_file, const TrainingData *sample, int sample_num) {
    FILE *log = fopen(log_file, "a");
    if (log) {
        fprintf(log, "Sample #%d:\n", sample_num);
        fprintf(log, "Label: %s\n", sample->label);
        fprintf(log, "Grid Data: ");
        for (int i = 0; i < INPUT_NODES; i++) {
            fprintf(log, "%d", sample->grid[i]);
            if (i < INPUT_NODES - 1) {
                fprintf(log, ",");
            }
        }
        fprintf(log, "\n");
        fclose(log);
    }
}

int load_training_data(const char *training_data, TrainingData **data, const char *log_file) {
    FILE *file = fopen(training_data, "r");
    if (!file) {
        log_message(log_file, "Failed to open training data file");
        return -1;
    }

    log_message(log_file, "Successfully opened training data file");

    int num_samples = 0; // Change this to an integer
    *data = malloc(sizeof(TrainingData) * 100); // Initial allocation for 100 samples
    if (*data == NULL) {
        log_message(log_file, "Failed to allocate initial memory");
        fclose(file);
        return -1;
    }

    log_message(log_file, "Allocated memory for 100 samples");

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        TrainingData sample;
        char *token = strtok(line, ",");
        if (token) {
            strncpy(sample.label, token, MAX_LABEL_LEN - 1);
            sample.label[MAX_LABEL_LEN - 1] = '\0'; // Null-terminate

            for (int i = 0; i < INPUT_NODES; i++) {
                token = strtok(NULL, ",");
                if (token) {
                    sample.grid[i] = atoi(token); // Convert to integer
                }
            }
            (*data)[num_samples++] = sample; // Use num_samples directly

            // Log each sample read
            log_sampletrainingdata(log_file, &sample, num_samples);

            // Reallocate if necessary
            if (num_samples % 100 == 0) {
                char log_buffer[128];
                snprintf(log_buffer, sizeof(log_buffer), "Reallocating memory for %d samples", num_samples + 100);
                log_message(log_file, log_buffer);

                *data = realloc(*data, sizeof(TrainingData) * (num_samples + 100));
                if (*data == NULL) {
                    log_message(log_file, "Failed to reallocate memory");
                    fclose(file);
                    return -1; // Handle reallocation failure
                }
            }
        }
    }

    // Log the total number of samples read
    char log_buffer[128];
    snprintf(log_buffer, sizeof(log_buffer), "Finished reading training data. Total samples: %d", num_samples);
    log_message(log_file, log_buffer);

    fclose(file);
    return num_samples; // Return number of samples loaded
}

// Function to log weights
void log_weights(const char *log_file, float weights[64][8]) {
    FILE *log = fopen(log_file, "a");
    if (log) {
        fprintf(log, "Weights:\n");
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES_LAYER1; j++) {
                fprintf(log, "%.6f ", weights[i][j]);
            }
            fprintf(log, "\n");
        }
        fclose(log);
    }
}

// Function to log biases
void log_biases(const char *log_file, float biases[8]) {
    FILE *log = fopen(log_file, "a");
    if (log) {
        fprintf(log, "Biases:\n");
        for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) {
            fprintf(log, "%.6f ", biases[i]);
        }
        fprintf(log, "\n");
        fclose(log);
    }
}

// Load weights from file with logging
void load_weights(const char *filename, float weights[64][8], const char *log_file) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open weights file");
        log_message(log_file, "Failed to open weights file");
        return;
    }

    log_message(log_file, "Opened weights file successfully");

    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++) {
            if (fscanf(file, "%f", &weights[i][j]) != 1) {
                log_message(log_file, "Error reading weights from file");
                fclose(file);
                return;
            }
        }
    }

    log_weights(log_file, weights); // Log the weights
    log_message(log_file, "Finished reading weights");
    fclose(file);
}

// Load biases from file with logging
void load_biases(const char *filename, float biases[8], const char *log_file) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open biases file");
        log_message(log_file, "Failed to open biases file");
        return;
    }

    log_message(log_file, "Opened biases file successfully");

    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) {
        if (fscanf(file, "%f", &biases[i]) != 1) {
            log_message(log_file, "Error reading biases from file");
            fclose(file);
            return;
        }
    }

    log_biases(log_file, biases); // Log the biases
    log_message(log_file, "Finished reading biases");
    fclose(file);
}

// Forward pass
void forward_pass(float input[INPUT_NODES], float output[OUTPUT_NODES],float weights1[64][8], float weights2[8][8], float weights3[8][2], float biases1[8], float biases2[8], float biases3[2]) {
    
    // Compute hidden1 (input to hidden1 layer)
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) {
        hidden1[i] = biases1[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            hidden1[i] += input[j] * weights1[j][i];
        }
        hidden1[i] = relu(hidden1[i]); // Apply activation
    }

    // Compute hidden2 (hidden1 to hidden2 layer)
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++) {
        hidden2[i] = biases2[i];
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++) {
            hidden2[i] += hidden1[j] * weights2[j][i];
        }
        hidden2[i] = relu(hidden2[i]); // Apply activation
    }

    // Compute output (hidden2 to output layer)
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output[i] = biases3[i];
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++) {
            output[i] += hidden2[j] * weights3[j][i];
        }
        output[i] = relu(output[i]); // Apply activation
    }
}

// Backward pass
void backward_pass(float output[OUTPUT_NODES], float input[INPUT_NODES], char *label, float weights1[64][8], float weights2[8][8], float weights3[8][2],float biases1[8], float biases2[8], float biases3[2]) {
    
    // Determine the expected output for the given label
    int expected_output = (strcmp(label, "two") == 0) ? 1 : 0;

    // Compute output error (gradient of loss w.r.t output)
    float output_error[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; ++i) {
        output_error[i] = (output[i] - (i == expected_output ? 1.0 : 0.0));
    }

    // Gradients for weights3 and biases3 (from hidden2 to output)
    float d_weights3[HIDDEN_NODES_LAYER2][OUTPUT_NODES] = {0}; // Derivatives for weights between hidden2 and output
    float d_biases3[OUTPUT_NODES] = {0}; // Derivatives for biases in the output layer

    for (int i = 0; i < OUTPUT_NODES; ++i) {
        d_biases3[i] = output_error[i]; // Bias gradient
        for (int j = 0; j < HIDDEN_NODES_LAYER2; ++j) {
            d_weights3[j][i] = output_error[i] * hidden2[j]; // Weight gradient
        }
    }

    // Backpropagate the error to hidden layer 2
    float hidden2_error[HIDDEN_NODES_LAYER2] = {0};
    for (int i = 0; i < HIDDEN_NODES_LAYER2; ++i) {
        for (int j = 0; j < OUTPUT_NODES; ++j) {
            hidden2_error[i] += output_error[j] * weights3[i][j];
        }
        hidden2_error[i] *= relu_derivative(hidden2[i]); // Multiply by derivative of activation function
    }

    // Gradients for weights2 and biases2 (from hidden1 to hidden2)
    float d_weights2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2] = {0}; // Derivatives for weights between hidden1 and hidden2
    float d_biases2[HIDDEN_NODES_LAYER2] = {0}; // Derivatives for biases in hidden layer 2

    for (int i = 0; i < HIDDEN_NODES_LAYER2; ++i) {
        d_biases2[i] = hidden2_error[i];
        for (int j = 0; j < HIDDEN_NODES_LAYER1; ++j) {
            d_weights2[j][i] = hidden2_error[i] * hidden1[j];
        }
    }

    // Backpropagate the error to hidden layer 1
    float hidden1_error[HIDDEN_NODES_LAYER1] = {0};
    for (int i = 0; i < HIDDEN_NODES_LAYER1; ++i) {
        for (int j = 0; j < HIDDEN_NODES_LAYER2; ++j) {
            hidden1_error[i] += hidden2_error[j] * weights2[i][j];
        }
        hidden1_error[i] *= relu_derivative(hidden1[i]); // Multiply by derivative of activation function
    }

    // Gradients for weights1 and biases1 (from input to hidden1)
    float d_weights1[INPUT_NODES][HIDDEN_NODES_LAYER1] = {0}; // Derivatives for weights between input and hidden1
    float d_biases1[HIDDEN_NODES_LAYER1] = {0}; // Derivatives for biases in hidden layer 1

    for (int i = 0; i < HIDDEN_NODES_LAYER1; ++i) {
        d_biases1[i] = hidden1_error[i];
        for (int j = 0; j < INPUT_NODES; ++j) {
            d_weights1[j][i] = hidden1_error[i] * input[j];
        }
    }
}

// Update weights and biases
void update_weights_and_biases(float weights1[64][8], float weights2[8][8], float weights3[8][2],
                                float biases1[8], float biases2[8], float biases3[2],
                                float d_weights1[64][8], float d_weights2[8][8], float d_weights3[8][2],
                                float d_biases1[8], float d_biases2[8], float d_biases3[2],
                                float learning_rate) {
    for (int i = 0; i < INPUT_NODES; ++i) {
        for (int j = 0; j < HIDDEN_NODES_LAYER1; ++j) {
            weights1[i][j] -= learning_rate * d_weights1[i][j];
        }
    }

    for (int i = 0; i < HIDDEN_NODES_LAYER1; ++i) {
        for (int j = 0; j < HIDDEN_NODES_LAYER2; ++j) {
            weights2[i][j] -= learning_rate * d_weights2[i][j];
        }
    }

    for (int i = 0; i < HIDDEN_NODES_LAYER2; ++i) {
        for (int j = 0; j < OUTPUT_NODES; ++j) {
            weights3[i][j] -= learning_rate * d_weights3[i][j];
        }
    }

    for (int i = 0; i < HIDDEN_NODES_LAYER1; ++i) {
        biases1[i] -= learning_rate * d_biases1[i];
    }

    for (int i = 0; i < HIDDEN_NODES_LAYER2; ++i) {
        biases2[i] -= learning_rate * d_biases2[i];
    }

    for (int i = 0; i < OUTPUT_NODES; ++i) {
        biases3[i] -= learning_rate * d_biases3[i];
    }
}

// Function to save weights to a file
void save_weights(float weights[][8], int rows, int cols, const char* filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f\n", weights[i][j]);
        }
    }
    fclose(file);
}

// Function to save biases to a file
void save_biases(float biases[], int size, const char* filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%f\n", biases[i]);
    }
    fclose(file);
}

// Function to save weights and biases for all layers
void save_weights_and_biases() {
    // Layer 1
    save_weights(Weights_Layer1, 64, 8, "Weights/weights1.txt");
    save_biases(Biases_Layer1, 8, "Biases/biases1.txt");

    // Layer 2
    save_weights(Weights_Layer2, 8, 8, "Weights/weights2.txt");
    save_biases(Biases_Layer2, 8, "Biases/biases2.txt");

    // Layer 3
    save_weights(Weights_Layer3, 8, 2, "Weights/weights3.txt");
    save_biases(Biases_Layer3, 2, "Biases/biases3.txt");
}

float compute_loss(float output[], char *label) {
    int target;
    
    // Convert label to target (1 if it's "two", 0 if "not two")
    if (strcmp(label, "two") == 0) {
        target = 1;
    } else {
        target = 0;
    }
    
    // Assuming output[1] represents the output for "two"
    // and output[0] represents "not two"
    float loss = -((target * log(output[1])) + ((1 - target) * log(output[0])));
    
    return loss;
}


// Train the neural network
void train(const char *training_data) {
    TrainingData *data;
    int num_samples = load_training_data(training_data, &data, log_filetrainingdata);
    if (num_samples <= 0) {
        printf("Failed to load training data\n");
        return;
    }

    for (int epoch = 0; epoch < 1000; epoch++) { // You can set your desired number of epochs
        float total_loss = 0.0f;

        for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
            TrainingData sample = data[sample_idx];

            // Forward pass
            forward_pass(sample.grid, output, Weights_Layer1, Weights_Layer2, Weights_Layer3, Biases_Layer1, Biases_Layer2, Biases_Layer3);

            // Backward pass
            backward_pass(output, sample.grid, sample.label, Weights_Layer1, Weights_Layer2, Weights_Layer3, Biases_Layer1, Biases_Layer2, Biases_Layer3);

            // Update weights and biases
            update_weights_and_biases(Weights_Layer1, Weights_Layer2, Weights_Layer3,
                                      Biases_Layer1, Biases_Layer2, Biases_Layer3,
                                      d_weights1, d_weights2, d_weights3,
                                      d_biases1, d_biases2, d_biases3, learning_rate);

            // Optionally, log loss or accuracy
            total_loss += compute_loss(output, sample.label); // Assuming you have a compute_loss function
        }

        // Log progress every few epochs
        if (epoch % 100 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, total_loss / num_samples);
        }
    }

    // Save final weights and biases after training
    save_weights_and_biases();

    // Free allocated memory for training data
    free(data);
}

