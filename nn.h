#ifndef nn_H
#define nn_H

//#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;


MatrixXf sigmoid(MatrixXf X);
MatrixXf dsigmoid(MatrixXf X);
MatrixXf softmax(MatrixXf X);
float nll(RowVectorXf yTrue, RowVectorXf yPred);

class NeuralNet{
//Méthodes
public:
	NeuralNet(int inputSize, int hiddenSize, int outputSize);
	MatrixXf forward(MatrixXf X);
	void forward_keep_activations(MatrixXf X, MatrixXf &y, MatrixXf &h, MatrixXf &zH);
	float loss(MatrixXf X, VectorXf y);
	~NeuralNet();

//Attributs	
public:
	MatrixXf wH;
	RowVectorXf bH;
	MatrixXf wO;
	float bO;
	int outputSize;	
};

#endif