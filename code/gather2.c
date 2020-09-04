#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void main(int argc, char *argv[])
{
    int* buffer;
    int* receive_counts;
    int* receive_displacements;
    int  rank, ntasks, i;
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    buffer = (int*)malloc(ntasks * sizeof(int));
    receive_counts = (int*)malloc(ntasks * sizeof(int));
    receive_displacements = (int*)malloc(ntasks * sizeof(int));
    
    for (i=0; i<ntasks; i++)
    {
        receive_counts[i] = 1;
        receive_displacements[i] = i;
    }
    
    MPI_Gatherv(&rank, 1, MPI_INT, buffer, receive_counts, receive_displacements, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
        printf("Results gathered: ");
        for (i=0; i<ntasks; i++)
            printf("%d ", buffer[i]);
        printf("\n");
    }
    
    MPI_Finalize();
    
    free(buffer);
    free(receive_counts);
    free(receive_displacements);
}
