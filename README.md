This is an implementation of a logistic regression and a one hidden layer neural network using CUDA.
In order to test the power of parallelization, we parallelized some dot products inside the algorithms
and tested the speed of the computation and the accuracy. Contact us by email at charles[dot]dognin[at]ensae[dot]fr or geremie[dot]djohossou[at]ensae[dot]fr to obtain the data. To test our parallelized algorithms we used the simplified CIFAR10 dataset, with two classes: cat or not cat. 

To run the models simply

```
git clone https://github.com/charlesdognin/Deep_Learning_GPU_CUDA
```

Use

```
cd Deep_Learning_GPU_CUDA
```

Obtain the data and place them in the Deep_Learning_GPU_CUDA file

Finally run:

```
nvcc regloss.cu -std=c++11 -arch=sm_50 -o dq
./dq
```

Or

```
nvcc neuralnet.cu -std=c++11 -arch=sm_50 -o dq
./dq
```
