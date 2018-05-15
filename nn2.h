#ifndef nn_H
#define nn_H

//#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;


RowVectorXf sigmoid(RowVectorXf X);
RowVectorXf dsigmoid(RowVectorXf X);
RowVectorXf softmax(RowVectorXf X);
float nll(RowVectorXf yTrue, RowVectorXf yPred);
void propagate(VectorXf w, float b, MatrixXf X, RowVectorXf Y, VectorXf &dw,  float &db, float &cost);

#endif
