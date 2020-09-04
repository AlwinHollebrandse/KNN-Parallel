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
#include <mpi.h>

using namespace std;

// A comparator function used by qsort 
int compare(const void * a, const void * b) { 
  const int *fa = *(const int **) a;
  const int *fb = *(const int **) b;
  return (fa[0] > fb[0]) - (fa[0] < fb[0]);
} 

// Performs majority voting using the first k first elements of an array
int kVoting(int k, float** shortestKDistances) { // TODO need k * number of processes? int len = sizeof(arr)/sizeof(arr[0]);
    map<float, int> classCounter;
    for (int i = 0; i < k; i++) {
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

// int* mainKNN(ArffData* dataset) {
//     printf("This system has %d processors configured and "
//         "%d processors available.\n",
//         get_nprocs_conf(), get_nprocs());
//     int elements_per_proc = dataset->num_instances() / get_nprocs_conf(); // TODO or get_nprocs()?\

//     int world_size, world_rank, sendcount, recvcount, source; // TODO is source needed?

//     MPI_Init(&argc,&argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

//     int world_rank = 0;
//     if (world_rank == 0) {

//     }

//     // Create a buffer that will hold a subset of the random numbers
//     float *sub_dataSet = (float*)malloc(sizeof(dataset->get_instance(0)) * elements_per_proc);

//     // Scatter the random numbers to all processes
//     MPI_Scatter(dataset, elements_per_proc, MPI_FLOAT, sub_dataSet,
//             elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD); // MPI_FLOAT should be the arff type? or a multiplier?

//     // Compute the average of your subset
//     float *shortestKDistances[k];
//     getKNNForInstance(dataset->get_instance(i), k, distancesAndClasses, shortestKDistances, sub_dataSet); // TODO k * number processes used
    
//     // Gather all partial averages down to the root process
//     float *sub_kNNs = NULL;

//     if (world_rank == 0) {
//         sub_kNNs = malloc(sizeof(float) * world_size);
//     }
//     MPI_Gather(&sub_kNNs, 1, MPI_FLOAT, sub_kNNs, 1, MPI_FLOAT, 0,
//             MPI_COMM_WORLD);

//     // Compute the total average of all numbers.
//     if (world_rank == 0) {
//         float avg = compute_avg(sub_avgs, world_size);
//         predictions[i] = kVoting(k, shortestKDistances);// TODO not using shortestDistances, but the combined of them
//     }
//     MPI_Finalize();
// }

float** getKNNForInstance(ArffInstance* instance, int k, float **distancesAndClasses, float **shortestKDistances, ArffData* dataset) { // TODO right type?
    int distancesAndClassesIndex = -1;

    // TODO parallel
    for(int j = 0; j < dataset->num_instances(); j++) { // target each other instance

        // TODO when this is in parallel, only 1 process will actual have the real data value. so only one needs this...
        // NOTE could share daset and just sent starting and ending indexes
        /// I think scatter gives a portion, so the above wont twork...
        // maybe a deep equal? does that work across processes? if instance was passed maybe
        // if (i == j) continue;
        if (instance == dataset->get_instance(j)) continue;

        distancesAndClassesIndex++;

        float *row = (float *)malloc(2 * sizeof(float));

        float distance = 0;
        
        for(int k = 0; k < dataset->num_attributes() - 1; k++) { // compute the distance between the two instances
            float diff = instance->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
            distance += diff * diff;
        }
        
        row[0] = sqrt(distance);
        row[1] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float();
        distancesAndClasses[distancesAndClassesIndex] = row;
    }

    qsort(distancesAndClasses, dataset->num_instances() - 1, (2 * sizeof(float)), compare);

    // TODO insert the MPI return here
    for(int j = 0; j < k; j++) {
        shortestKDistances[j] = distancesAndClasses[j];
    }
}

int* KNN(ArffData* dataset) {
    printf("This system has %d processors configured and "
        "%d processors available.\n",
        get_nprocs_conf(), get_nprocs());
    int elements_per_proc = dataset->num_instances() / get_nprocs_conf(); // TODO or get_nprocs()?

    int world_size, world_rank, sendcount, recvcount, source; // TODO is source needed? or most of these

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    int k = 1;
    if (k > dataset->num_instances() - 1)
        k = dataset->num_instances() - 1;

    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    // TODO parallel
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance in the dataset
        float *distancesAndClasses[dataset->num_instances() - 1]; // NOTE the -1 is there so that the instance in question wont be included

        // serial **************************************************************
        // float *shortestKDistances[k];
        // getKNNForInstance(dataset->get_instance(i), k, distancesAndClasses, shortestKDistances, dataset); // TODO k * number processes used


        // inner parallel **************************************************************
        // Create a buffer that will hold a subset of the random numbers
        float *sub_dataSet = (float*)malloc(sizeof(dataset->get_instance(0)) * elements_per_proc);

        // Scatter the random numbers to all processes
        MPI_Scatter(dataset, elements_per_proc, MPI_FLOAT, sub_dataSet,
                elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD); // MPI_FLOAT should be the arff type? or a multiplier?

        // Compute the average of your subset
        float *shortestKDistances[k];
        getKNNForInstance(dataset->get_instance(i), k, distancesAndClasses, shortestKDistances, sub_dataSet); // TODO k * number processes used
        
        // Gather all partial averages down to the root process
        float *sub_kNNs = NULL;

        if (world_rank == 0) {
            sub_kNNs = malloc(sizeof(float) * world_size);
        }
        MPI_Gather(&sub_kNNs, 1, MPI_FLOAT, sub_kNNs, 1, MPI_FLOAT, 0,
                MPI_COMM_WORLD);

        // Compute the total average of all numbers.
        if (world_rank == 0) {
            predictions[i] = kVoting(k, shortestKDistances);// TODO not using shortestDistances, but the combined of them
        }
        // MPI_Finalize();


        predictions[i] = kVoting(k, shortestKDistances);

        // TODO move free
        for (int j = 0; j < dataset->num_instances() - 1; j++) {
            free(distancesAndClasses[j]);
        }
        
    }
    MPI_Finalize();
    return predictions;
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
    int* predictions = KNN(dataset);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    free(predictions);
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
