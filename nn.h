#ifndef nn_H
#define nn_H

//#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;


RowVectorXf sigmoid(RowVectorXf X);
RowVectorXf dsigmoid(RowVectorXf X);
RowVectorXf softmax(RowVectorXf X);
float nll(RowVectorXf yTrue, RowVectorXf yPred);

class NeuralNet{
//Méthodes
public:
	NeuralNet(int inputSize, int hiddenSize, int outputSize);
	RowVectorXf forward(RowVectorXf x);
	void NeuralNet::forward_keep_activations(RowVectorXf X, RowVectorXf &y, RowVectorXf &z, RowVectorXf &zH);
	~NeuralNet();

//Attributs	
public:
	MatrixXf wH;
	RowVectorXf bH;
	MatrixXf wO;
	RowVectorXf bO;
	int outputSize;	
};

#endif