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
#define N 512
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

// To load the data in the right format
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
void initialize(MatrixXf &w1, MatrixXf &w2, VectorXf &b1, float &b2, int dim_x, int dim_h, int dim_y){
	w1 = MatrixXf::Random(dim_h, dim_x).matrix();
  b1 = ArrayXf::Zero(dim_h).matrix();
  w2 = MatrixXf::Random(dim_y, dim_h).matrix();
  b2 = 0;
}
//float y_i, VectorXf &dw, float &db, float &cost)
void fwd_propagate_i(VectorXf X_i, MatrixXf w1, MatrixXf w2, VectorXf b1, float b2, VectorXf &a1_i, VectorXf &z1_i,  VectorXf &z2_i, float &a2_i){
   z1_i = (w1 * X_i).matrix() + b1.matrix();
   a1_i = sigmoid(z1_i);
   z2_i = (w2 * a1_i).array() + b2;
   a2_i = sigmoid_i(z2_i(0));
 }

// Compute the cost for one point
void compute_cost_i(float y_i, float a2_i, float &cost){
   float temp = log(a2_i) * y_i + log((1.0 - a2_i)) * (1.0 - y_i);
   cost = - 1.0 * temp;}

// Backward Propagation for one point
void bwd_propagation_i(VectorXf X_i, float y_i, float a2_i, MatrixXf w1, MatrixXf w2, VectorXf &a1_i, MatrixXf &dw1, MatrixXf &dw2, VectorXf &db1, float &db2){
    float dz2_i = a2_i - y_i;
    dw2 = 1.0 * (dz2_i * a1_i.transpose()).matrix();
    db2 = 1.0 * dz2_i;
    VectorXf dz1_i = ((w2.transpose() * dz2_i).array() * (1 - a1_i.array().pow(2)).array()).matrix();
    dw1 = 1.0 * (dz1_i * X_i.transpose());
    //Rq sum or not?
    db1 = 1.0 * dz1_i;
}

// Forward Propagation for one point
void fwd_propagate_i_par(VectorXf X_i, MatrixXf w1, MatrixXf w2, VectorXf b1, float b2, VectorXf &a1_i, VectorXf &z1_i,  VectorXf &z2_i, float &a2_i){

  float *a, *b, *c;
  float *dev_a, *dev_b, *dev_c;
  float size = N * sizeof(float);

 //allocate space for the variables on the device
  cudaMalloc(&dev_a, size);
  cudaMalloc(&dev_b, size);
  cudaMalloc(&dev_c, sizeof(float));

  //allocate space for the variables on the host
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(sizeof(float));


  dev_a = w1.data();
  dev_b = X_i.data();

  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
  cudaMemset(dev_c, 0.0f, sizeof(float));

  dot_par<<< N, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

  a1_i = sigmoid(*c + b1.array());

  free(a);
  free(b);
  free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  z2_i = (w2 * a1_i).array() + b2;
  a2_i = sigmoid_i(z2_i(0));
 }

// Update parameters for optimization
void update_parameters(MatrixXf &w1, MatrixXf &w2, VectorXf &b1, float &b2, MatrixXf dw1, MatrixXf dw2,
  VectorXf &db1, float &db2, float learningRate, int j){
    w1 = w1 - ((learningRate / sqrt(j + 1)) * dw1);
    b1 = b1 - ((learningRate / sqrt(j + 1)) * db1);
    w2 = w2 - ((learningRate / sqrt(j + 1)) * dw2);
    b2 = b2 - ((learningRate / sqrt(j + 1)) * db2);
}

// Prediction Function
RowVectorXf predict(MatrixXf x, MatrixXf w1, MatrixXf w2, VectorXf b1, float b2){
	int m = x.cols();
  VectorXf a1_i, z1_i, z2_i;
  float a2_i;
	RowVectorXf yPrediction(m);
	for(int i = 0; i < m; i++){
    fwd_propagate_i(x.col(i), w1, w2, b1, b2, a1_i, z1_i, z2_i, a2_i);
		if(a2_i <= 0.5){
			yPrediction(0, i) = 0;
		}
		else{
			yPrediction(0, i) = 1;
		}
	}
	return(yPrediction);
}

// Main model
void model(MatrixXf xTrain, MatrixXf yTrain, MatrixXf xTest, MatrixXf yTest,
  RowVectorXf &yPredictionsTrain, RowVectorXf &yPredictionsTest,
  VectorXf db1, float db2, MatrixXf dw1, MatrixXf dw2, int numIterations,
  float learningRate,  int dim_x, int dim_h, int dim_y, bool par=true){
  int m = xTrain.cols();
  float cost;
  MatrixXf w1, w2;
  VectorXf b1, a1_i, z1_i, z2_i;
  float b2,  a2_i;
  initialize(w1, w2, b1, b2, dim_x, dim_h, dim_y);
	for(int j = 0; j < numIterations; j++){
  		random_device rd;
      mt19937 gen(rd());
      uniform_int_distribution<int> dis(0, m - 1);
      int i = dis(gen);
      if (par==true){
        fwd_propagate_i_par(xTrain.col(i), w1, w2, b1, b2, a1_i, z1_i, z2_i, a2_i);
        compute_cost_i(yTrain(i), a2_i, cost);
        bwd_propagation_i(xTrain.col(i), yTrain(i), a2_i, w1, w2, a1_i, dw1, dw2, db1, db2);
        update_parameters(w1, w2, b1, b2, dw1, dw2, db1, db2, learningRate, j);
      }
      else{
        fwd_propagate_i(xTrain.col(i), w1, w2, b1, b2, a1_i, z1_i, z2_i, a2_i);
        compute_cost_i(yTrain(i), a2_i, cost);
        bwd_propagation_i(xTrain.col(i), yTrain(i), a2_i, w1, w2, a1_i, dw1, dw2, db1, db2);
        update_parameters(w1, w2, b1, b2, dw1, dw2, db1, db2, learningRate, j);
      }
		if((j % 100) == 0){
            cout << "Cost after epoch " << j << ": " << cost << endl;}}

  yPredictionsTrain = predict(xTrain, w1, w2, b1, b2);
  yPredictionsTest = predict(xTest, w1, w2, b1, b2);
  cout << "train accuracy: " << 100 - ((yPredictionsTrain - yTrain).array().abs().sum() / float(yTrain.size())) * 100 << endl;
  cout << "test accuracy: " << 100 - ((yPredictionsTest - yTest).array().abs().sum() / float(yTest.size())) * 100 << endl;
}

int main(){
  Timer Tim1, Tim2;
	MatrixXf w1, w2, dw1, dw2;
	VectorXf b1, db1, z1_i, a1_i(4), z2_i;
  float db2;

  MatrixXf xTrain = load_csv("trainingImages.csv") / 255.0;
	RowVectorXf yTrain = load_csv("trainingLabels.csv");
	MatrixXf xTest = load_csv("testImages.csv") / 255.0;
	RowVectorXf yTest = load_csv("testLabels.csv");
	std::cout << "x train: " << xTrain.rows() << " " << xTrain.cols() << std::endl;
	std::cout << "y train: " << yTrain.rows() << " " << yTrain.cols() << std::endl;
	std::cout << "x test: " << xTest.rows() << " " << xTest.cols() << std::endl;
	std::cout << "y test: " << yTest.rows() << " " << yTest.cols() << std::endl;
	RowVectorXf yPredictionsTrain, yPredictionsTest;

  cout << "Warming the GPU..." << endl;
  model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, db1, db2, dw1, dw2, 500, 0.1, xTrain.rows(), 10, 1, false);

  Tim1.start();
  model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, db1, db2, dw1, dw2, 1000, 0.1, xTrain.rows(), 30, 1, false);
  Tim1.add();
  cout << "With SGD CPU Time is: " << Tim1.getsum() << " seconds" << endl;

  Tim2.start();
  model(xTrain, yTrain, xTest, yTest, yPredictionsTrain, yPredictionsTest, db1, db2, dw1, dw2, 1000, 0.1, xTrain.rows(), 30, 1);
  Tim2.add();
	cout << "With SGD GPU Time is: " << Tim2.getsum() << " seconds" << endl;

	return(0);
}
