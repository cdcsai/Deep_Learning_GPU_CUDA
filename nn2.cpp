#include <iostream>
#include "Eigen/Dense"
#include "nn.h"

using namespace Eigen;
using namespace std;


void initialization(VectorXf &w, float &b, int dim){
	w = VectorXf::Random(dim, 1) * 0.01;
	b = 0;
}

void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf Y, VectorXf &dw,  float &db, float &cost){
	int m = X.rows();

	// Forward Propagation
	RowVectorXf A;
	A = sigmoid((w.transpose()*X).array() + b).matrix();
	ArrayXf A1 = (Y * A.array().log().matrix()).array();
	A2 = (1 - Y.array()) * (1 - A.array()).log()
	cost = (- 1 / m) * (A1 + A2).sum();

	// Back Propagation
	dw = (1./m)*np.dot(X,(A-Y).T);
  db = (1./m)*np.sum(A-Y);

	return dw, db, cost

}

void optimize(w, b, X, Y, itr, lr){
	costs = [];

	for i in range(itr){
		dw, db, cost = propagate(VectorXf w, float b, MatrixXf X, RowVectorXf Y, VectorXf &dw,  float &db, float &cost);
		w = w - lr * dw;
		b = b - lr * db;
	}
	return w, b, dw, db

}




//This defines the sigmoid function
RowVectorXf sigmoid(RowVectorXf X){
	ArrayXf expo = (-X.transpose()).array().exp();
	ArrayXf result = 1 / (1 + expo);
	return(result.matrix().transpose());
}

// This defines the stochastic gradient descent
// RowVectorXf sgd(RowVectorXf theta, int size, int nitr){
//   int i;
//   for (int i=0; i<nitr; i++){
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_int_distribution<int> dis(1, size);
//     i = dis(gen);
//     theta = theta - this -> stepsize * grad_i(theta, i);
//   }
//   return theta;
// }

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

// //This defines the negative likelihood
// float nll(RowVectorXf yTrue, RowVectorXf yPred){
// 	VectorXf logLikelihoods;
// 	ArrayXf epsilon = ArrayXf::Constant(yPred.transpose().size(), 1, 0.0001); //0.0001 is chosen to avoid infty in log value
// 	logLikelihoods = ((yPred.transpose().array() + epsilon).log() * yTrue.transpose().array()).colwise().sum().matrix();
// 	//cout << logLikelihoods.rows() << ", " << logLikelihoods.cols() << endl;
// 	float result;
// 	result = -logLikelihoods.sum() / logLikelihoods.size();
// 	return(result);
// }

// RowVectorXf forward(RowVectorXf X){
// 	RowVectorXf h;
// 	RowVectorXf y;
// 	h = sigmoid(X * wH + bH);
// 	y = softmax(h * wO + bO);
// 	return(y);
// }


// void forward_keep_activations(RowVectorXf X, RowVectorXf &y, RowVectorXf &h, RowVectorXf &zH){
// 	zH = X * this -> wH + bH;
// 	h = sigmoid(zH);
// 	RowVector zO = h * this -> wO + this -> bO;
// 	y = softmax(zO);
// }

void model(X, Y, itr, lr){

	w,b = initialization();
	w, b, dw, db, cost = optimize(w, b, X, Y, itr, lr);

	return cost;


}

int main()
{
  VectorXf w(3);
	w << 1,
				2,
				3;
  float b = 2;
	MatrixXf X(3, 3);
	X << 1, 2, 3,
				1, 2, 3,
				1, 2, 3;
	RowVectorXf Y(3);
	Y << 1, 2, 3;
	VectorXf dw;
	float db;
	float cost;
	propagate(w, b, X, Y, dw, db, cost);
	return 0;
}
