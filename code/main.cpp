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
int kVoting(int globalK, float** shortestKDistances) { // TODO need globalK * number of processes? int len = sizeof(arr)/sizeof(arr[0]);
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

    // for (auto i : classCounter) {
    //     printf("class: %d, numberOfVotes: %d ", (int)i.first, i.second);
    // }

    return voteResult;
}

// void printArray(float *datasetAsMatrix, int numberOfRows, int numberOfCols, int rank) {
//     for (int x = 0; x < numberOfRows; x++) {
//         for (int y = 0; y < numberOfCols; y++) {
//             printf("rank:%d, datasetAsMatrix[%d][%d]: %f\n", rank, x, y, *(datasetAsMatrix + x * numberOfCols + y));
//         }
//         printf("\n");
//     }
// }

float** getKNNForInstance(int i, int k, float **distancesAndClasses, float **shortestKDistances, ArffData* dataset) {
    // TODO witht he outer loop setup, you only enter this if the i is within your endingIndex - startingIndex
    // printf("in getKNNForInstance for rank: %d, i: %d, startingIndex: %d, endingIndex: %d\n", rank, i, startingIndex, endingIndex);
    int distancesAndClassesIndex = -1;

    for(int j = 0; j < dataset->num_instances(); j++) { // target each other instance
        if (i == j) continue;

        distancesAndClassesIndex++;
        float *row = (float *)malloc(2 * sizeof(float)); // TODO realloc?
        float distance = 0;

        for(int k = 0; k < dataset->num_attributes() - 1; k++) { // compute the distance between the two instances
            float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
            distance += diff * diff;
        }

        row[0] = sqrt(distance);
        row[1] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float();
        distancesAndClasses[distancesAndClassesIndex] = row;
    }

    qsort(distancesAndClasses, dataset->num_instances() - 1, (2 * sizeof(float)), compare); // TODO dont need to sort, needd to find "k" shortest

    for(int j = 0; j < k; j++) {
        shortestKDistances[j] = distancesAndClasses[j];
    }
}

int* KNN(ArffData* dataset, int argc, char *argv[]) {
    printf("This system has %d processors configured and "
        "%d processors available.\n",
        get_nprocs_conf(), get_nprocs());

    // int world_size, world_rank, sendcount, recvcount, source; // TODO is source needed? or most of these

    // TODO what can be deleted?
    int SIZE = 2;
    int numProcesses, rank, sendcount, recvcount, source;
    float sendbuf[SIZE][SIZE] = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}  };
    float recvbuf[SIZE];

    int globalK = 3; // TODO make a command line arg
    if (globalK > dataset->num_instances() - 1) // NOTE the - 1 is needed because you cant compare to yourself
        globalK = dataset->num_instances() - 1;

    MPI_Init(&argc,&argv); // TODO past this point, till finalize, its all private?...but then why does the processor info print -np times?
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    printf("Hello, world.  I am %d of %d\n", rank, numProcesses);
    
    source = 0;

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

    printf("rank: %d, startingIndex: %d, endingIndex: %d, theoreticalEndingIndex - startingIndex: %d\n", rank, startingIndex, endingIndex, theoreticalEndingIndex - startingIndex);

    // Compute the kNN of a processes's data
    float *distancesAndClasses[dataset->num_instances() - 1];

    // Need to ensure that the 'k' value is never bigger than the amount of data a processes will go through
    float *shortestKDistances[globalK];

    int *processPredictions = (int*)malloc((theoreticalEndingIndex - startingIndex) * sizeof(int));
    int processPredictionsIndex = -1;

    for(int i = startingIndex; i < endingIndex; i++) { // for each instance in the dataset that the processes has the indexes for

        processPredictionsIndex++;
        getKNNForInstance(i, globalK, distancesAndClasses, shortestKDistances, dataset);
        processPredictions[processPredictionsIndex] = kVoting(globalK, shortestKDistances);
        printf("rank: %d, i: %d, predicted class: %d\n", rank, i, processPredictions[processPredictionsIndex]);

        // TODO move free, and have all needed frees
        // for (int j = 0; j < dataset->num_instances() - 1; j++) {
        //     free(allProcessesShortestKDistances[j]);
        // }
    }

    int *finalPredictions = NULL;
    if (rank == source) 
        finalPredictions = (int*)malloc(sizeof(int) * (theoreticalEndingIndex - startingIndex) * numProcesses);
    MPI_Gather(processPredictions, theoreticalEndingIndex - startingIndex, MPI_INTEGER, finalPredictions, theoreticalEndingIndex - startingIndex, MPI_INTEGER, source, MPI_COMM_WORLD);
    if (rank == source) {
        printf("\nfinalPredictions with potential excess: ");
        for (int x = 0; x < (theoreticalEndingIndex - startingIndex) * numProcesses; x++) {
            printf("%d, ", finalPredictions[x]);
        }
        printf("\nfinalPredictions: ");
        for (int x = 0; x < dataset->num_instances(); x++) {
            printf("%d, ", finalPredictions[x]);
        }
    }
    

    // printf("rank: %d is done\n", rank);
    MPI_Finalize(); // TODO move this and init to main?
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
    if(argc != 2) {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    int* predictions = KNN(dataset, argc, argv);
    // Compute the confusion matrix
    // int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // // Calculate the accuracy
    // float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    // free(predictions);
  
    // printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
