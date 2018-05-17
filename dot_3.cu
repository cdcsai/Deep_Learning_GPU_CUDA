#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

using namespace std;

__global__ void dot(int *aa, int *bb, int *cc)
{
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = aa[index] * bb[index];

    __syncthreads();

    if (0 == threadIdx.x)
    {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        atomicAdd(cc, sum);
    }
}

int main()
{
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c(0);
    int size = N * sizeof(int);

   //allocate space for the variables on the device
    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, sizeof(int));

   //allocate space for the variables on the host
   a = (int *)malloc(size);
   b = (int *)malloc(size);
   c = (int *)malloc(sizeof(int));

   //this is our ground truth
   int sumTest = 0;
   //generate numbers
   for (int i = 0; i < N; i++)
   {
       //a[i] = rand() % 10;
       //b[i] = rand() % 10;
	   a[i] = i;
	   b[i] = 10;
       sumTest += a[i] * b[i];
       //printf(" %d %d \n",a[i],b[i]);
   }

   //*c = 0;

   cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);

   dot<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);

   cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
   cout << "c: " << *c << endl;
   cout << "sumTest: " << sumTest << endl;
   //printf("%i ", *c);
   //printf("%d ", sumTest);

   free(a);
   free(b);
   free(c);

   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFree(dev_c);
   return 0;
 }