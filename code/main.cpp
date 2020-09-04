#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <map>

using namespace std;


// A comparator function used by qsort 
int compare(const void * a, const void * b) { 
  const int *fa = *(const int **) a;
  const int *fb = *(const int **) b;
  return (fa[0] > fb[0]) - (fa[0] < fb[0]);
} 

// Performs majority voting using the first k first elements of an array
int kVoting(int k, float** shortestKDistances) {
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

int* KNN(ArffData* dataset)
{
    int k = 1;
    if (k > dataset->num_instances() - 1)
        k = dataset->num_instances() - 1;

    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance in the dataset
    {
        float *distancesAndClasses[dataset->num_instances() - 1]; // NOTE the -1 is there so that the instance in question wont be included
        int distancesAndClassesIndex = -1;

        for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
        {
            if (i == j) continue;

            distancesAndClassesIndex++;

            float *row = (float *)malloc(2 * sizeof(float));

            float distance = 0;
            
            for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
            {
                float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                distance += diff * diff;
            }
            
            row[0] = sqrt(distance);
            row[1] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator float();
            distancesAndClasses[distancesAndClassesIndex] = row;
        }

        qsort(distancesAndClasses, dataset->num_instances() - 1, (2 * sizeof(float)), compare); // TODO is the 3 arg correct? or shouldit be a pointer?

        // TODO insert the MPI return here
        // TODO add frees for all added mallocs?
        float *shortestKDistances[k];
        for(int j = 0; j < k; j++) {
            shortestKDistances[j] = distancesAndClasses[j];
        }
        
        predictions[i] = kVoting(k, shortestKDistances);

        for (int j = 0; j < dataset->num_instances() - 1; j++) {
            free(distancesAndClasses[j]);
        }
    }
    
    return predictions;
}

int* KNNOpenMP(ArffData* dataset)
{
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    #pragma omp parallel for
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance in the dataset
    {
        float smallestDistance = FLT_MAX;
        int smallestDistanceClass;

        #pragma omp parallel for
        for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
        {
            if(i == j) continue;
            
            float distance = 0;
            
            for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
            {
                float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                distance += diff * diff;
            }
            
            distance = sqrt(distance);
            
            if(distance < smallestDistance) // select the closest one
            {
                smallestDistance = distance;
                smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
            }
        }
        
        predictions[i] = smallestDistanceClass;
    }
    
    return predictions;
}


int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    uint64_t diff;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    int* predictions = KNN(dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);
  
    printf("The 1NN classifier sequential for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    predictions = KNNOpenMP(dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    
    confusionMatrix = computeConfusionMatrix(predictions, dataset);
    accuracy = computeAccuracy(confusionMatrix, dataset);
  
    printf("The 1NN classifier OpenMP for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);    
}
