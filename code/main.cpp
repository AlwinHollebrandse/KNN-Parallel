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

// TODO make distancesAndClasses variable size? one needs a -1
float** getKNNForInstance(int i, int k, float **distancesAndClasses, float **shortestKDistances, ArffData* dataset, int startingIndex, int endingIndex, int distancesAndClassesSize) {
    int distancesAndClassesIndex = -1;

    for(int j = startingIndex; j < endingIndex; j++) { // target each other instance
        if (i == j) continue;

        distancesAndClassesIndex++;
        float *row = (float *)malloc(2 * sizeof(float));
        float distance = 0;

        for(int k = 0; k < dataset->num_attributes() - 1; k++) { // compute the distance between the two instances
            float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
            distance += diff * diff;
        }

        row[0] = sqrt(distance);
        row[1] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float();
        distancesAndClasses[distancesAndClassesIndex] = row;
    }

    for (int x = 0; x < distancesAndClassesSize; x++) {
        printf("distancesAndClasses[%d]: distance: %f, class: %f\n", x, distancesAndClasses[x][0], distancesAndClasses[x][1]);
    }

    qsort(distancesAndClasses, distancesAndClassesSize, (2 * sizeof(float)), compare);

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
    int numtasks, rank, sendcount, recvcount, source;
    float sendbuf[SIZE][SIZE] = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}  };
    float recvbuf[SIZE];

    int globalK = 3; // TODO make a command line arg
    if (globalK > dataset->num_instances() - 1)
        globalK = dataset->num_instances() - 1;

    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    MPI_Init(&argc,&argv); // TODO past this point, till finalize, its all private?...but then why does the processor info print -np times?
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    printf("Hello, world.  I am %d of %d\n", rank, numtasks);

    // TODO parallel
    for(int i = 0; i < 1; i++) {//dataset->num_instances(); i++) { // for each instance in the dataset
        // printf("threads exist? rank: %d, i: %d", rank, i);
        
        // Scatter the random numbers to all processes
        int floats_per_proc =  (dataset->num_instances() * dataset->num_attributes()) / numtasks; // (dataset->num_attributes() * dataset->num_instances()) / numtasks; // TODO am I sending a dataset elem or each float of it? is * dataset->num_attributes();
        int instances_per_proc = dataset->num_instances() / numtasks;
        if (dataset->num_instances() % numtasks > 0)
            instances_per_proc++;

        // TODO technocially, this could be outside the outside loop. Both inner and outer would use it.
        int *allThreadInnerIndexesArray = (int*)malloc(sizeof(int) * numtasks * 2);
        if (rank == 0) {
            printf("instances_per_proc: %d\n", instances_per_proc);

            for (int currentIndex = 0; currentIndex  < 2 * numtasks; currentIndex++) {
                int threadStartingIndex = 0;
                if (currentIndex > 0)
                    threadStartingIndex = allThreadInnerIndexesArray[currentIndex - 1];
                allThreadInnerIndexesArray[currentIndex] = threadStartingIndex;
                printf("threadStartingIndex: %d\n", threadStartingIndex);

                currentIndex++;

                int threadEndingIndex = allThreadInnerIndexesArray[currentIndex - 1] + instances_per_proc;
                printf("threadEndingIndex: %d\n", threadEndingIndex);
                if (threadEndingIndex >= dataset->num_instances())
                    threadEndingIndex = dataset->num_instances(); // set to max possible
                allThreadInnerIndexesArray[currentIndex] = threadEndingIndex;
            }

            printf("allThreadInnerIndexesArray: ");
            for (int x = 0; x < numtasks * 2; x++)
                printf("%d, ", allThreadInnerIndexesArray[x]);
            printf("\n");
        }

        // printArray (datasetAsMatrix, dataset->num_instances(), dataset->num_attributes(), rank);

        // printf("scatter\n");

        // define source task and elements to send/receive, then perform collective scatter
        source = 0; // TODO technocially, this could be outside the outside loop. Both inner and outer would use it.
        int *threadInnerIndexesArray = (int*)malloc(sizeof(int) * 2);
        MPI_Scatter(allThreadInnerIndexesArray, 2, MPI_INTEGER, threadInnerIndexesArray, 
            2, MPI_INTEGER, source, MPI_COMM_WORLD);

            // printArray(sub_dataSet, instances_per_proc, dataset->num_attributes(), rank);
                    // TODO free subarray
        int startingIndex = threadInnerIndexesArray[0];
        int endingIndex = threadInnerIndexesArray[1];
        printf("rank: %d, threadInnerIndexesArray: %d, %d\n", rank, startingIndex, endingIndex);

        // Compute the kNN of a thread's data
        int localK = globalK;
        if (localK > instances_per_proc - 1)
            localK = instances_per_proc - 1;

        int distancesAndClassesSize = endingIndex - startingIndex;
        if (i >= startingIndex && i < endingIndex)
            distancesAndClassesSize--;
        float *distancesAndClasses[distancesAndClassesSize];
        // float *distancesAndClasses[endingIndex - startingIndex];// - 1]; // NOTE the -1 is there so that the instance in question wont be included // TODO is this a shared or private variable
        float *shortestKDistances[localK];
        // TODO dataset will be changed to non arff type after outer mpi scatter
        
        int number;
        if (rank == 0) {
            printf("rank: %d, distancesAndClassesSize: %d\n", rank, distancesAndClassesSize);

            getKNNForInstance(i, localK, distancesAndClasses, shortestKDistances, dataset, startingIndex, endingIndex, distancesAndClassesSize);
            // printf("rank: %d, instance: ", rank, dataset->get_instance(i)
            for (int x = 0; x < localK; x++)
                printf("rank: %d, shortestKDistances: %d, distance: %f, class: %f\n", rank, x, shortestKDistances[x][0], shortestKDistances[x][1]);
            number = -1;

            MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            printf("\n\n");
        }  else if (rank == 1) {
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("rank: %d, distancesAndClassesSize: %d\n", rank, distancesAndClassesSize);

            getKNNForInstance(i, localK, distancesAndClasses, shortestKDistances, dataset, startingIndex, endingIndex, distancesAndClassesSize);
            // printf("rank: %d, instance: ", rank, dataset->get_instance(i)
            for (int x = 0; x < localK; x++)
                printf("rank: %d, shortestKDistances: %d, distance: %f, class: %f\n", rank, x, shortestKDistances[x][0], shortestKDistances[x][1]);
        }
        


        
        // printf("hi2");

        // // Gather all partial averages down to the root process
        // float *sub_kNNs = NULL;

        // if (rank == 0) {
        //     sub_kNNs = (float*)malloc(sizeof(float) * floats_per_proc * globalK); // TODO ont be correct size. since its a 2d array that also includes size...
        // }
        // MPI_Gather(&sub_kNNs, 1, MPI_FLOAT, sub_kNNs, 1, MPI_FLOAT, 0,
        //         MPI_COMM_WORLD);

        // // Compute the total average of all numbers.
        // if (world_rank == 0) {
        //     qsort(distancesAndClasses, dataset->num_instances() - 1, (2 * sizeof(float)), compare);
        //     predictions[i] = kVoting(globalK, shortestKDistances);// TODO not using shortestDistances, but the combined of them
        // }
        // MPI_Finalize();


        // predictions[i] = kVoting(globalK, shortestKDistances);

        // TODO move free
        // for (int j = 0; j < dataset->num_instances() - 1; j++) {
        //     free(distancesAndClasses[j]);
        // }
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
