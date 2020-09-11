#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <map>
#include <sys/sysinfo.h>
#include "mpi.h"

using namespace std;

// A comparator function used by qsort 
int compare(const void * a, const void * b) { 
  const int *fa = *(const int **) a;
  const int *fb = *(const int **) b;
  return (fa[0] > fb[0]) - (fa[0] < fb[0]);
} 

// Performs majority voting using the first globalK first elements of an array
int kVoting(int globalK, float** shortestKDistances) {
    // for (int i =0; i < globalK; i++) {
    //     printf("i: %d, distance: %f, class: %f\n", i, shortestKDistances[i][0], shortestKDistances[i][1]);
    // }

    map<float, int> classCounter;
    for (int i = 0; i < globalK; i++) {
        classCounter[shortestKDistances[i][1]]++;
    }

    int voteResult = -1;
    int numberOfVotes = -1;
    for (auto i : classCounter) {
        if (i.second > numberOfVotes) {
            numberOfVotes = i.second;
            voteResult = i.first;
        }
    }

    return voteResult;
}

float** getKNNForInstance(int i, int k, float **distancesAndClasses, float **shortestKDistances, ArffData* dataset) {
    int distancesAndClassesIndex = 0;

    for(int j = 0; j < dataset->num_instances(); j++) { // target each other instance
        if (i == j) continue;

        float *row = (float *)malloc(2 * sizeof(float)); // TODO realloc? // TODO free
        float distance = 0;

        for(int k = 0; k < dataset->num_attributes() - 1; k++) { // compute the distance between the two instances
            float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
            distance += diff * diff;
        }

        row[0] = sqrt(distance);
        row[1] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float();
        // float row[] = {sqrt(distance), dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float()};
        distancesAndClasses[distancesAndClassesIndex] = row;
        distancesAndClassesIndex++;
    }

    qsort(distancesAndClasses, dataset->num_instances() - 1, (2 * sizeof(float)), compare); // TODO dont need to sort, need to find "k" shortest

    printf("shortestKDistances: ");
    for(int j = 0; j < k; j++) {
        shortestKDistances[j] = distancesAndClasses[j];
        printf("row: %d, distance: %f, class: %f\n", j, shortestKDistances[j][0], shortestKDistances[j][1]);
    }
}

int* KNN(ArffData* dataset, int rank, int numProcesses, int source, int globalK) {
    int instances_per_proc = dataset->num_instances() / numProcesses;
    if (dataset->num_instances() % numProcesses > 0)
        instances_per_proc++;

    int startingIndex = rank * instances_per_proc;
    int theoreticalEndingIndex = (rank + 1) * instances_per_proc; // NOTE its theoretical because there might be excess values that arent there at the end TODO is this true? check
    int endingIndex;
    if (theoreticalEndingIndex >= dataset->num_instances())
        endingIndex = dataset->num_instances(); // set to max possible
    else
        endingIndex = theoreticalEndingIndex;

    // Compute the kNN of a processes's data
    float *distancesAndClasses[dataset->num_instances() - 1];

    // Need to ensure that the 'k' value is never bigger than the amount of data a processes will go through
    float *shortestKDistances[globalK];

    int predictionSize = theoreticalEndingIndex - startingIndex;

    printf("rank: %d, startingIndex: %d, endingIndex: %d, predictionSize: %d\n", rank, startingIndex, endingIndex, predictionSize);

    int *processPredictions = (int*)malloc(predictionSize * sizeof(int));
    int processPredictionsIndex = 0;

    for(int i = startingIndex; i < endingIndex; i++) { // for each instance in the dataset that the processes has the indexes for
    // for(int i = 0; i < 1; i++) { // for each instance in the dataset that the processes has the indexes for

        getKNNForInstance(i, globalK, distancesAndClasses, shortestKDistances, dataset);
        processPredictions[processPredictionsIndex] = kVoting(globalK, shortestKDistances);
        // printf("rank: %d, i: %d, predicted class: %d\n", rank, i, processPredictions[processPredictionsIndex]);
        processPredictionsIndex++;
    }

    int *finalPredictions = NULL;
    if (rank == source) 
        finalPredictions = (int*)malloc(sizeof(int) * predictionSize * numProcesses);

    MPI_Gather(processPredictions, predictionSize, MPI_INTEGER, finalPredictions, predictionSize, MPI_INTEGER, source, MPI_COMM_WORLD);
    if (rank == source) {
        printf("\nfinalPredictions with potential excess: ");
        for (int x = 0; x < predictionSize * numProcesses; x++) {
            printf("%d, ", finalPredictions[x]);
        }
        printf("\nfinalPredictions: ");
        for (int x = 0; x < dataset->num_instances(); x++) {
            printf("%d, ", finalPredictions[x]);
        }
    }

    free(processPredictions);
    // for (int j = 0; j < predictionSize; j++) {
    //     free(processPredictions[j]);
    // }
    return finalPredictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset) {
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset) {
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        cout << "Usage: ./main datasets/datasetFile.arff kValue" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    printf("This system has %d processors configured and "
        "%d processors available.\n",
        get_nprocs_conf(), get_nprocs());

    int rank, numProcesses, source = 0;
    int globalK = atoi(argv[2]);
    if (globalK > dataset->num_instances() - 1) // NOTE the - 1 is needed because you cant compare to yourself
        globalK = dataset->num_instances() - 1;

    MPI_Init(&argc,&argv); // TODO past this point, till finalize, its all private?...but then why does the processor info print -np times?
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    printf("Hello, world.  I am %d of %d\n", rank, numProcesses);
    
    // Get the class predictions
    // int* predictions = KNN(dataset, argc, argv);
    int* predictions = KNN(dataset, rank, numProcesses, source, globalK);
    if (rank == source) {
        for(int i = 0; i < dataset->num_instances(); i++) {
            printf("%d, ", predictions[i]);
        }
    }

    if (rank == source) { // TODO move init and finalize so rank can be used here?
        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, dataset);
        
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;


        printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
    }

    free(predictions);
    MPI_Finalize();
}
