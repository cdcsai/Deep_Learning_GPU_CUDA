#include <iostream>
#include </usr/users/hpcgif/hpcgif_9/Deep_Learning_GPU_CUDA/Eigen/Dense>
#include <vector>
#include <math.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#define N 512
#define THREADS_PER_BLOCK 512

using namespace Eigen;
using namespace std;

// The Cuda parallelized dot product
__global__ void dot_par(float *aa, float *bb, float *cc)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = aa[index] * bb[index];

    __syncthreads();

    if (0 == threadIdx.x)
    {
        float sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        atomicAdd(cc, sum);
    }
}

//This defines the sigmoid function
MatrixXf sigmoid(MatrixXf X){
	ArrayXXf expo = (-X).array().exp();
	ArrayXXf result = 1 / (1 + expo);
	return(result.matrix());
}

float sigmoid_i(float X){
	float result = 1 / (1 + exp(-X));
	return(result);
}

//This defines the dsigmoid function
MatrixXf dsigmoid(MatrixXf X){
	ArrayXXf sig = sigmoid(X).array();
	ArrayXXf result = sig * (1 - sig);
	return(result.matrix());
}

//This function defines the softmax function
MatrixXf softmax(MatrixXf X){
	ArrayXXf e = X.array().exp();
	ArrayXf result = e / e.sum();
	return (result.matrix());
}

//This function initializes the coefficient
void initialize(VectorXf &w, float &b, int dim){
	w = ArrayXf::Random(dim).matrix();
	b = 0;
}

void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
	int m = X.cols();
	MatrixXf A = sigmoid((w.transpose() * X).array() + b);
	cost = (-1. / m) * (((y.array() * A.array().log()) + ((1 - y.array()) * (1 - A.array()).log())).sum());
	dw = (1. / m) * (X * ((A - y).transpose()));
	db = (1. / m) * ((A - y).sum());
}

void propagate_i(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost){
	float a_i = sigmoid_i(w.dot(X_i) + b);
	cost = -1 * ((y_i * log(a_i)) + ((1 - y_i) * log(1 - a_i)));
	dw = (X_i * (a_i - y_i));
	db = a_i - y_i;
}

void propagate_i_par(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost, bool partial=false){

  float *a, *b1, *c;
  float *dev_a, *dev_b, *dev_c(0);
  float size = N * sizeof(float);

 //allocate space for the variables on the device
  cudaMalloc(&dev_a, size);
  cudaMalloc(&dev_b, size);
  cudaMalloc(&dev_c, sizeof(float));

 //allocate space for the variables on the host
 a = (float *)malloc(size);
 b1 = (float *)malloc(size);
 c = (float *)malloc(sizeof(float));

 dev_a = w.transpose().data();
 dev_b = X_i.data();

 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b1, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_c, c, sizeof(float), cudaMemcpyHostToDevice);

 dot_par<<< N, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);

 cudaMemcpy(c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

 float a_i = sigmoid_i(*c + b);

 free(a);
 free(b1);
 free(c);
 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);

  cost = -1 * ((y_i * log(a_i)) + ((1 - y_i) * log(1 - a_i)));
  dw = (X_i * (a_i - y_i));
  db = a_i - y_i;
 }

void propagate_par(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
  int m = X.cols();
  VectorXf dw_a;
  float db_a;
  for (int i=0; i<m; i++){
    float cost_i;
    propagate_i_par(w, b, X.col(i), y(i), dw, db, cost_i);
    cost += cost_i;
    dw_a += dw;
    db_a += db;
  }
  dw = (1 / m) * dw_a;
  db = (1 / m) * db_a;
  cost = (1 / m) * cost;
}

void optimize(VectorXf &w, float &b, VectorXf &dw, float &db, MatrixXf X, RowVectorXf y,
			  int numIterations, float learningRate, vector<float> &costs, bool printCost=true, bool par=false, bool sgd=false){
	int m = X.cols();
  float cost;
	for(int j = 0; j < numIterations; j++){
    if (sgd == true){
  		random_device rd;
      mt19937 gen(rd());
      uniform_int_distribution<int> dis(0, m - 1);
      int i = dis(gen);

        if (par == true){
          propagate_i_par(w, b, X.col(i), y(i), dw, db, cost);
        }
        else{
          propagate_i(w, b, X.col(i), y(i), dw, db, cost);
      }}
    else{

      if (par == true){
        propagate_par(w, b, X, y, dw, db, cost);
      }
      else{
        propagate(w, b, X, y, dw, db, cost);
    }}

		w = w - ((learningRate / sqrt(j + 1)) * dw);
		b = b - ((learningRate / sqrt(j + 1)) * db);
		if (j % 100 == 0){
			costs.push_back(cost);
		}
		if(printCost and (j % 1000) == 0)
            cout << "Cost after iteration " << j << ": " << cost << endl;
	}
}

RowVectorXf predict(VectorXf w, float b, MatrixXf X){
	int m = X.cols();
	RowVectorXf yPrediction(m);

	MatrixXf A = sigmoid((w.transpose() * X).array() + b);
	for(int i = 0; i < A.cols(); i++){
		if(A(0, i) <= 0.5){
			yPrediction(0, i) = 0;
		}
		else{
			yPrediction(0, i) = 1;
		}
	}
	return(yPrediction);
}

void model(MatrixXf xTrain, RowVectorXf yTrain, MatrixXf xTest, RowVectorXf yTest, RowVectorXf &yPredictionsTrain,
           RowVectorXf &yPredictionsTest, VectorXf &w, float &b, std::vector<float> &costs, const int &numIterations, const float &learningRate,
		   bool printCost = true, bool par=true, bool sgd=false){
	initialize(w, b, xTrain.rows());
	VectorXf dw;
	float db;
  if (sgd==true){
    if (par==true){
      optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, par=true, sgd=true);
    }
    else{
      optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, sgd=true);
    }
  }
  else{
    if (par==true){
      optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, par=true);
    }
    else{
      optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs);
  }
	yPredictionsTrain = predict(w, b, xTrain);
	yPredictionsTest = predict(w, b, xTest);

	cout << "train accuracy: " << 100 - ((yPredictionsTrain - yTrain).array().abs().sum() / float(yTrain.size())) * 100 << endl;
	cout << "test accuracy: " << 100 - ((yPredictionsTest - yTest).array().abs().sum() / float(yTest.size())) * 100 << endl;
}}

int main(){
  Timer Tim;
	VectorXf w, dw;
	float b, db, cost;
	initialize(w, b, 4);
	MatrixXf x(4, 26);
	x << 1, 0, 3, 4, 1, 2, 2, 3, 4, 2, 3, 5, 1, 2, 3, 2, 5, 0, 1, 4, 5, 0, 1, 2, 1, 3,
       2, 5, 3, 5, 2, 1, 4, 3, 2, 0, 0, 2, 4, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1, 2, 5,
	   2, 3, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1, 2, 5, 3, 5, 2, 1, 4, 3, 2, 0, 0, 2, 4,
	   2, 3, 2, 5, 0, 1, 4, 5, 0, 1, 2, 1, 3, 2, 3, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1;
	MatrixXf xTest(4, 5);
	xTest << 1, 3, 4, 5, 3,
	         0, 4, 5, 4, 4,
			 2, 3, 2, 1, 1,
			 3, 4, 0, 0, 5;
	RowVectorXf y(26), yTest(5);
	yTest << 1, 1, 1, 0, 0;
	RowVectorXf yPredictions, yPredictionsTest;
	y << 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0;
	vector<float> costs;
  Tim.start();
	model(x, y, xTest, yTest, yPredictions, yPredictionsTest, w, b, costs, 10000, 0.0001);
  Tim.add();
	cout << "With GPU Time is: " << Tim.getsum() << " seconds" << endl;

  Tim.start();
  model(x, y, xTest, yTest, yPredictions, yPredictionsTest, w, b, costs, 10000, 0.0001, true, false, false);
  Tim.add();
  cout << "With CPU Time is: " << Tim.getsum() << " seconds" << endl;

	return(0);
}
