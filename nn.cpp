#include <iostream>
#include <Eigen/Dense>
#include "nn.h"

using namespace Eigen;
using namespace std;


//This defines the sigmoid function
RowVectorXf sigmoid(RowVectorXf X){
	ArrayXf expo = (-X.transpose()).array().exp();
	ArrayXf result = 1 / (1 + expo);
	return(result.matrix().transpose());
}

//This defines the dsigmoid function
RowVectorXf dsigmoid(RowVectorXf X){
	ArrayXf sig = sigmoid(X).transpose().array();
	ArrayXf result = sig * (1 - sig);
	return(result.matrix().transpose());
}

//This function defines the softmax function
RowVectorXf softmax(RowVectorXf X){
	ArrayXf e = X.transpose().array().exp();
	ArrayXf result = e / e.sum();
	return (result.matrix().transpose());
}

//This defines the negative likelihood
float nll(RowVectorXf yTrue, RowVectorXf yPred){
	VectorXf logLikelihoods;
	ArrayXf epsilon = ArrayXf::Constant(yPred.transpose().size(), 1, 0.0001); //0.0001 is chosen to avoid infty in log value
	logLikelihoods = ((yPred.transpose().array() + epsilon).log() * yTrue.transpose().array()).colwise().sum().matrix();
	//cout << logLikelihoods.rows() << ", " << logLikelihoods.cols() << endl;
	float result;
	result = -logLikelihoods.sum() / logLikelihoods.size();
	return(result);
}

//Constructor of NeuralNet class
NeuralNet::NeuralNet(int inputSize, int hiddenSize, int outputSize):
wH(MatrixXf::Random(inputSize, hiddenSize) * 0.01), bH(ArrayXf::Zero(hiddenSize).matrix().transpose()), 
wO(MatrixXf::Random(hiddenSize, outputSize) * 0.01), bO(ArrayXf::Zero(outputSize).matrix().transpose()), outputSize(outputSize)
{
	
}

//Destructor of NeuralNet class
NeuralNet::~NeuralNet(){
	
}


RowVectorXf NeuralNet::forward(RowVectorXf X){
	RowVectorXf h;
	RowVectorXf y;
	h = sigmoid(X * this -> wH + this -> bH);
	y = softmax(h * this -> wO + this -> bO);
	return(y);
}


void NeuralNet::forward_keep_activations(RowVectorXf X, RowVectorXf &y, RowVectorXf &z, RowVectorXf &zH){
	RowVectorXf h;
}

int main()
{
  RowVectorXf yTrue(7), yPred(7);
  RowVectorXf x(7), res(7);
  float result;
  yTrue << 1, 0, 0, 0, 1, 1, 1;
  yPred << 0, 1, 0, 0, 1, 1, 1;
  x << 1, 2, 4, 2, 3, 0, 5;
  NeuralNet nn(7, 5, 2);
  cout << "****wH************" << endl;
  cout << nn.wH << endl;
  cout << "****wO************" << endl;
  cout << nn.wO << endl;
  cout << "****bH************" << endl;
  cout << nn.bH << endl;
  cout << "****bO************" << endl;
  cout << nn.bO << endl;
  res = nn.forward(x);
  cout << "****forward************" << endl;
  cout << res << endl;
  return 0;
}
