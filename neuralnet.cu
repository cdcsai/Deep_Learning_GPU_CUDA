#include <iostream>
#include </usr/users/hpcgif/hpcgif_9/Deep_Learning_GPU_CUDA/Eigen/Dense>
#include <vector>
#include <math.h>
#include <random>

using namespace Eigen;
using namespace std;



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

// CUDA functions used for training the NN
__global__ void dot_par(int*ret, RowVectorXf *w, VectorXf *X){
  float temp[26];
  temp[threadIdx.x] = w[threadIdx.x] * X[threadIdx.x];
  __syncthreads();
  if (0 == threadIdx.x){
    int sum = 0;
    for (int i=0; i<26; i++){
      sum += temp[i];
    ret* = sum;
    }
  }}

// Prop and back prop parallelised using the dot product
void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
	int m = X.cols();
  int d = X.rows();

  int *ret;
  cudaMallocManaged(&ret, d * sizeof(float));
  MatrixXf A = sigmoid((dot_par<<< 1, 1 >>>(ret, w.transpose(), X.rows(3)) + b));
      cudaDeviceSynchronize();
  cudaFree(ret);
	cost = (-1. / m) * (((y.array() * A.array().log()) + ((1 - y.array()) * (1 - A.array()).log())).sum());
	dw = (1. / m) * (X * ((A - y).transpose()));
	db = (1. / m) * ((A - y).sum());}

//__device__ __managed__  int  ret[1000];
//__global__ void AplusB(int a, int b) {
  //  ret[threadIdx.x] = a + b + threadIdx.x;

void propagate_i(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost_i){
	float a_i = sigmoid_i(w.dot(X_i) + b);
	cost_i = -1 * ((y_i * log(a_i)) + ((1 - y_i) * log(1 - a_i)));
	dw = (X_i * (a_i - y_i));
	db = a_i - y_i;
}

void optimize(VectorXf &w, float &b, VectorXf &dw, float &db, MatrixXf X, RowVectorXf y,
			  int numIterations, float learningRate, vector<float> &costs, bool printCost = true){
	int m = X.cols();
	for(int j = 0; j < numIterations; j++){
		random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, m - 1);
    int i = dis(gen);
		float cost;
		propagate(w, b, X, y, dw, db, cost);
		w = w - ((learningRate / sqrt(j + 1)) * dw);
		b = b - ((learningRate / sqrt(j + 1)) * db);
		if (i % 100 == 0){
			costs.push_back(cost);
		}
		if(printCost and (j % 10) == 0)
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
		   bool printCost = true){
	initialize(w, b, xTrain.rows());
	VectorXf dw;
	float db;
	optimize(w, b, dw, db, xTrain, yTrain, numIterations, learningRate, costs);
	yPredictionsTrain = predict(w, b, xTrain);
	yPredictionsTest = predict(w, b, xTest);

	cout << "train accuracy: " << 100 - ((yPredictionsTrain - yTrain).array().abs().sum() / float(yTrain.size())) << endl;
	cout << "test accuracy: " << 100 - ((yPredictionsTest - yTest).array().abs().sum() / float(yTest.size())) << endl;
}

int main(){
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
	//vector<float> costs;
	propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost);

	//propagate_i(w, b, x.col(3), y(3), dw, db, cost_i);
	//model(x, y, xTest, yTest, yPredictions, yPredictionsTest, w, b, costs, 10000, 0.0002);

	return(0);
}
