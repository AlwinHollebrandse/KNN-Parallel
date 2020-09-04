#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void main(int argc, char *argv[])
{
    int* buffer_send;
    int* buffer_recv;
    int  rank, ntasks, i;
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    buffer_send = (int*)malloc(sizeof(int));
    buffer_recv = (int*)malloc(ntasks * sizeof(int));
    
    buffer_send[0] = rank;
    
    MPI_Gather(buffer_send, 1, MPI_INT, buffer_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
        printf("Results gathered: ");
        for (i=0; i<ntasks; i++)
            printf("%d ", buffer_recv[i]);
        printf("\n");
    }
    
    MPI_Finalize();
    
    free(buffer_send);
    free(buffer_recv);
}
