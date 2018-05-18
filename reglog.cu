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
#include <fstream>
#include <stdlib.h>
#define N 2048
#define THREADS_PER_BLOCK 512

using namespace Eigen;
using namespace std;


// The Cuda parallelized dot product
__global__ void dot_par(float *aa, float *bb, float *cc)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[THREADS_PER_BLOCK];
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

// Fonction to load the data
MatrixXf load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<float> values;
        int rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Map<const Matrix<typename MatrixXf::Scalar, MatrixXf::RowsAtCompileTime, MatrixXf::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
    }

//This defines the sigmoid function
MatrixXf sigmoid(MatrixXf X){
	ArrayXXf expo = (-X).array().exp();
	ArrayXXf result = 1 / (1 + expo);
	return(result.matrix());
}

//This defines the sigmoid function for one point
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

// Propagation for the logistic regression
void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
	int m = X.cols();
	MatrixXf A = sigmoid((w.transpose() * X).array() + b);
	cost = (-1. / m) * (((y.array() * A.array().log()) + ((1 - y.array()) * (1 - A.array()).log())).sum());
	dw = (1. / m) * (X * ((A - y).transpose()));
	db = (1. / m) * ((A - y).sum());
}

// Propagation for the logistic regression for one point
void propagate_i(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost){
	float a_i = sigmoid_i(w.dot(X_i) + b);
	cost = -1 * ((y_i * log(a_i)) + ((1 - y_i) * log(1 - a_i)));
	dw = (X_i * (a_i - y_i));
	db = a_i - y_i;
}

// Parallelized Propagation for the logistic regression for one point
void propagate_i_par(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost){

  float *a, *b1, *c;
  float *dev_a, *dev_b, *dev_c;
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
 cudaMemset(dev_c, 0.0f, sizeof(float));

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

// Propagation for the logistic regression for the whole gradient
void propagate_par(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
  int m = X.cols();
  int d = X.rows();
  VectorXf dw_a(0);
  float db_a=0, cost_a=0;
  for (int i=0; i<m; i++){
    propagate_i_par(w, b, X.col(i), y(i), dw, db, cost);
    cost_a += cost;
    dw_a += dw;
    db_a += db;}
  dw = (1.0 / m) * dw_a;
  db = (1.0 / m) * db_a;

  cost = (1.0 / m) * cost_a;

}

// Optimize function
void optimize(VectorXf &w, float &b, VectorXf &dw, float &db, MatrixXf X, RowVectorXf y,
			  int numIterations, float learningRate, vector<float> &costs,
        bool par=false, bool sgd=false, bool printcost=true){
	int m = X.cols();
  float cost;
	for(int j = 0; j < numIterations; j++){
    if (sgd == true){
  		random_device rd;
      mt19937 gen(rd());
      uniform_int_distribution<int> dis(0, m - 1);
      int i = dis(gen);
      if (par == true){
        propagate_i_par(w, b, X.col(i), y(i), dw, db, cost);}

      else{
          propagate_i(w, b, X.col(i), y(i), dw, db, cost);
      }}

    else{
      if (par == true){
        propagate_par(w, b, X, y, dw, db, cost);}
      else{
          propagate(w, b, X, y, dw, db, cost);
    }}

		  w = w - ((learningRate / sqrt(j + 1)) * dw);
      b = b - ((learningRate / sqrt(j + 1)) * db);
		if (j % 100 == 0){
			costs.push_back(cost);
		}
		if((j % 1000) == 0 && printcost==true){
            cout << "Cost after iteration " << j << ": " << cost << endl;}
	}
}

// Function used for the prediction
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

// Main Model gathering all the functions
void model(MatrixXf xTrain, RowVectorXf yTrain, MatrixXf xTest, RowVectorXf yTest, RowVectorXf &yPredictionsTrain,
           RowVectorXf &yPredictionsTest, VectorXf &w, float &b, std::vector<float> &costs, const int &numIterations, const float &learningRate,
           bool par, bool sgd){
	initialize(w, b, xTrain.rows());
	VectorXf dw;
	float db;
  if (par==true &&  sgd==true){
    optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, true, true);}
  if (par==true &&  sgd==false){
    optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, true, false);}
  if (par==false && sgd==true){
    optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, false, true);}
  if (par==false && sgd==false){
      optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs, false, false);}

	  yPredictionsTrain = predict(w, b, xTrain);
	  yPredictionsTest = predict(w, b, xTest);
    cout << "train accuracy: " << 100 - ((yPredictionsTrain - yTrain).array().abs().sum() / float(yTrain.size())) * 100 << endl;
    cout << "test accuracy: " << 100 - ((yPredictionsTest - yTest).array().abs().sum() / float(yTest.size())) * 100 << endl;
}

int main(){
  Timer Tim1, Tim2;
	VectorXf w, dw;
	float b;
  MatrixXf xTrain = load_csv("trainingImages.csv") / 255.0;
	RowVectorXf yTrain = load_csv("trainingLabels.csv");
	MatrixXf xTest = load_csv("testImages.csv") / 255.0;
	RowVectorXf yTest = load_csv("testLabels.csv");
	std::cout << "x train: " << xTrain.rows() << " " << xTrain.cols() << std::endl;
	std::cout << "y train: " << yTrain.rows() << " " << yTrain.cols() << std::endl;
	std::cout << "x test: " << xTest.rows() << " " << xTest.cols() << std::endl;
	std::cout << "y test: " << yTest.rows() << " " << yTest.cols() << std::endl;
	RowVectorXf yPredictionsTrain, yPredictionsTest;
  vector<float> costs;

  cout << "Warming the GPU..." << endl;
  model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, w, b, costs, 1000, 0.1, true, true);

  Tim1.start();
	model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, w, b, costs, 10000, 0.01, true, true);
  Tim1.add();
	cout << "With SGD GPU Time is: " << Tim1.getsum() << " seconds" << endl;

  Tim2.start();
  model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, w, b, costs, 10000, 0.01, false, true);
  Tim2.add();
	cout << "With SGD CPU Time is: " << Tim2.getsum() << " seconds" << endl;

	return(0);
}
