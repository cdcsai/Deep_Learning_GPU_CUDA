#include <stdio.h>
#include <iostream>
#include <Eigen/Dense> //The Eigen library for linear algebra
using namespace std;
using namespace Eigen;
typedef Matrix<long, Dynamic, 1> VectorXl;



// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}



// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


__global__ void kernel_vect(long *a, long *b, long *c, long L){
	long idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx<L){
		c[idx] = a[idx] + b[idx];
	}
}

__global__ void kernel_vect_big(long *a, long *b, long *c, long L){
	long idx = threadIdx.x + blockIdx.x*blockDim.x;
	while(idx < L){
		c[idx] = a[idx] + b[idx];
		idx += blockDim.x + gridDim.x;
	}
}

int main(){
	int count;
	cudaDeviceProp prop;
	testCUDA(cudaGetDeviceCount(&count));
	testCUDA(cudaGetDeviceProperties(&prop, count - 1));
	long L = 160000000;
	// Déclaration des vecteurs Eigen
	VectorXl aEigen(L);
	VectorXl bEigen(L);
	VectorXl cEigen(L);
	//Initialisation des vecteurs Eigen
	for (long i = 0; i < L; i++){
		aEigen(i) = 2 * (i + 1);
		bEigen(i) = 3 * (i + 1);
		cEigen(i) = 1; //initialisation du vecteur c à 1
	}
	
	//Déclaration et initialisation des pointeurs sur int, en utilisant les vecteurs Eigen
	long *a, *b, *c;
	a = aEigen.data();
	b = bEigen.data();
	c = cEigen.data();
	
	cout << "****** BEFORE ************" << endl;
	cout << "a: " << a[10] << endl;
	cout << "b: " << b[10] << endl;
	cout << "c: " << c[10] << endl;
	cout << "--------------" << endl;
	cout << "a: " << a[150] << endl;
	cout << "b: " << b[150] << endl;
	cout << "c: " << c[150] << endl;
	cout << "--------------" << endl;
	cout << "a: " << a[2000] << endl;
	cout << "b: " << b[2000] << endl;
	cout << "c: " << c[2000] << endl;

	//Déclaration des vecteurs GPU, allocation de mémoire et initialisation par recopie
	long *aGPU, *bGPU, *cGPU;
	
	//Memory allocation on the GPU
	testCUDA(cudaMalloc(&aGPU, L * sizeof(long)));
	testCUDA(cudaMalloc(&bGPU, L * sizeof(long)));
	testCUDA(cudaMalloc(&cGPU, L * sizeof(long)));
	testCUDA(cudaMemcpy(aGPU, a, L * sizeof(long), cudaMemcpyHostToDevice));
	testCUDA(cudaMemcpy(bGPU, b, L * sizeof(long), cudaMemcpyHostToDevice));
	
	//Launching the operation on the GPU
	
	if((L+1024-1)/1024 < prop.maxGridSize[0]){
		cout << "no while loop needed " << endl;
		kernel_vect<<<(L+1024-1)/1024, 1024>>>(aGPU, bGPU, cGPU, L);
	}
	else{
		cout << "while loop needed " << endl;
		kernel_vect_big<<<1024, 1024>>>(aGPU, bGPU, cGPU, L);
	}
	
	//Copy the value from one ProcUnit to the other ProcUnit
	testCUDA(cudaMemcpy(c, cGPU, L * sizeof(long), cudaMemcpyDeviceToHost));
	
	//Freeing the GPU memory
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));

	cout << "****** AFTER ************" << endl;
	cout << "a: " << a[10] << endl;
	cout << "b: " << b[10] << endl;
	cout << "c: " << c[10] << endl;
	cout << "--------------" << endl;
	cout << "a: " << a[150000000] << endl;
	cout << "b: " << b[150000000] << endl;
	cout << "c: " << c[150000000] << endl;
	cout << "--------------" << endl;
	cout << "a: " << a[2000] << endl;
	cout << "b: " << b[2000] << endl;
	cout << "c: " << c[2000] << endl;
	
	/*
	free(a);
	free(b);
	free(c);
	*/
	return 0;
}