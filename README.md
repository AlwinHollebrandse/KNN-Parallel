# ParallelKNN
A KNN classifer in both serial and parallel versions. Written in C++.

NOTE: this kNN classifier predicts each data element in the provided dataset by using the remainder of the dataset.
NOTE: If the provided k is larger than the number of elements in the dataset, it gets set to the size of the dataset.  

## Running:
A terminal with MPI installed is required to run this code. Provided that requirement is met, run `make` in the terminal and then
`mpiexec -np NUMBEROFPROCESSORS ./main DATASET KVALUE` where NUMBEROFPROCESSORS and KVALUE are integers provided by the user. DATASET
is an arff file provided by the user. Examples of a proper arff file can be found under `datasets/`.

### Sample Input:
mpiexec -np 16 ./main ./datasets/large.arff 5

### Sample Output:
The KNN classifier for 19592 instances required 13840 ms CPU time, accuracy was 1.0000

## Implementation Details:
The current final iteration of the parallel version of this code works works as follows. Each MPI process gets a copy of the dataset and then computes its own "starting" and "ending" index of the dataset. Essentially, these indices exist to divide the dataset evenly among the existing processes by ensuring that only one process will find the knn of a given dataset instance. Once that is completed, each process will loop through its own data instances and compute the knn by utilizing the rest of the dataset, including the index values outside the process's data "chunk" (as defined by its starting and ending indices). Each of these data points results in a single prediction, ie. the result of the kNN. These results are placed into an array. After the data "chunk" has been completed, the source process (currently hard coded to be rank 0) will create a "finalPredicitons" array and gather all process' "chunk" of predictions into it, including its own prediction "chunk". This "finalPredicitons" array is then returned from the kNN method and compared to the dataset's actual classes to compute and report a final accuracy. In addition to reporting the accuracy, this code also reports the number of instances in the dataset and the amount of CPU time the kNN classifier took. The CPU time is computed by using "clock_gettime" functionality before and after the kNN call.

## kNN:
The kNN call (getKNNForInstance) is computed as follows: for a given dataset instance, calculate the euclidean distance compared to each other dataset instances. Record these distances and the attached class of each value. Sort this resulting array in ascending order according to distance. Take the first `k` pairs of the sorted list and perform `kVoting` to get a prediction. Voting works but getting the count of each class in the kNN values. The class with the most votes is predicted. In the event of a tie, the first class encountered that had that vote amount is returned.

## Results:
the following times were computed by averaging the run CPU times (in ms) for each of the provided dataset 3 times with a k of value 5.

| -np: | large | medium | small |
| --- | --- | --- | --- |
| 1 | 169452.3333 | 10587.66667 | 266 |
| 2 | 88465.33333 | 5428 | 256.6666667 |
| 4 | 19592 | 2941.333333 | 263.3333333 |
| 8 | TODO | 1678.666667 | 267 |
| 16 | 13622.33333 | 1081 | 295.3333333 |
