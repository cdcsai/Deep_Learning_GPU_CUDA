#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "nn.h"

using namespace Eigen;
using namespace std;


//This defines the sigmoid function
MatrixXf sigmoid(MatrixXf X){
	ArrayXXf expo = (-X).array().exp();
	ArrayXXf result = 1 / (1 + expo);
	return(result.matrix());
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
	w = ArrayXf::Zero(dim).matrix();
	b = 0;
}

void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost){
	int m = X.cols();
	MatrixXf A = sigmoid((w.transpose() * X).array() + b); 
	cost = (-1. / m) * (((y.array() * A.array().log()) + ((1 - y.array()) * (1 - A.array()).log())).sum());
	dw = (1. / m) * (X * ((A - y).transpose()));
	db = (1. / m) * ((A - y).sum());
}

void optimize(VectorXf &w, float &b, VectorXf &dw, float &db, MatrixXf X, RowVectorXf y, 
			  int numIterations, float learningRate, std::vector<float> &costs, bool printCost = true){
	for(int i = 0; i < numIterations; i++){
		float cost;
		propagate(w, b, X, y, dw, db, cost);
		w = w - (learningRate * dw);
		b = b - (learningRate * db);
		if (i % 10 == 0){
			costs.push_back(cost);
		}
		if(printCost and (i % 10) == 0)
            cout << "Cost after iteration " << i << ": " << cost << endl;
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
	std::vector<float> costs;
	model(x, y, xTest, yTest, yPredictions, yPredictionsTest, w, b, costs, 500, 0.01);
	return(0);	
}