#ifndef neuralnet_H
#define neuralnet_H

//#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std

MatrixXf sigmoid(MatrixXf X);
MatrixXf dsigmoid(MatrixXf X);
float sigmoid_i(float X);
MatrixXf softmax(MatrixXf X);
void initialize(VectorXf &w, float &b, int dim);
void propagate(VectorXf &w, float &b, MatrixXf &X, RowVectorXf &y, VectorXf &dw, float &db, float &cost);
void propagate_i(VectorXf w, float b, VectorXf X_i, float y_i, VectorXf &dw, float &db, float &cost_i);
void optimize(VectorXf &w, float &b, VectorXf &dw, float &db, MatrixXf X, RowVectorXf y,
			  int numIterations, float learningRate, std::vector<float> &costs, bool printCost = true);
RowVectorXf predict(VectorXf w, float b, MatrixXf X);
void model(MatrixXf xTrain, RowVectorXf yTrain, MatrixXf xTest, RowVectorXf yTest, RowVectorXf &yPredictionsTrain,
           RowVectorXf &yPredictionsTest, VectorXf &w, float &b, std::vector<float> &costs, const int &numIterations, const float &learningRate,
		   bool printCost = true);
#endif
