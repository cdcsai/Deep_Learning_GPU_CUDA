#include <iostream>
#include <Eigen/Dense>
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
wO(MatrixXf::Random(hiddenSize, outputSize) * 0.01), bO(0), outputSize(outputSize)
{
	
}

//Destructor of NeuralNet class
NeuralNet::~NeuralNet(){
	
}


MatrixXf NeuralNet::forward(MatrixXf X){
	MatrixXf h;
	MatrixXf y;
	h = sigmoid((X * this -> wH).rowwise() + (this -> bH));
	y = softmax(((h * this -> wO).array() + this -> bO).matrix());
	return(y);
}


void NeuralNet::forward_keep_activations(MatrixXf X, MatrixXf &y, MatrixXf &h, MatrixXf &zH){
	zH = (X * this -> wH).rowwise() + this -> bH;
	h = sigmoid(zH);
	MatrixXf zO = ((h * this -> wO).array() + this -> bO).matrix();
	y = softmax(zO);
}


float NeuralNet::loss(MatrixXf X, VectorXf y){
		return(nll(y.transpose(), (this -> forward)(X).transpose()));
}

int main()
{
  /*
  RowVectorXf yTrue(7), yPred(7);
  RowVectorXf x(7), res(7);
  float result;
  yTrue << 1, 0, 0, 0, 1, 1, 1;
  yPred << 0, 1, 0, 0, 1, 1, 1;
  x << 1, 2, 4, 2, 3, 0, 5;
  */
  NeuralNet nn(4, 3, 1);
  MatrixXf X(6, 4);
  VectorXf y(6);
  y << 0, 
          1, 
		  1, 
		  1,
		  0,
		  0;
  MatrixXf res, h, zH;
  X << 10, 0, 3, 4,
       20, 25, 3, 5,
	   2, 3, 0, 5,
	   2, 3, 12, 5,
	   2, 3, 12, 5,
	   0, 0, 0, 1;
  float result;
  /*
  cout << "wH" << endl;
  cout << nn.wH << endl;
  cout << "bH" << endl;
  cout << nn.bH << endl;
  cout << "wO" << endl;
  cout << nn.wO << endl;
  cout << "bO" << endl;
  cout << nn.bO << endl;
  */
  result = nn.loss(X, y);
  cout << "****loss************" << endl;
  cout << result << endl;
  return 0;
}
