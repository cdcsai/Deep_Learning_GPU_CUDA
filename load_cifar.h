#ifndef load_cifar_H
#define load_cifar_H

//#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;


template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v);
struct processedData {
  VectorXi trainingImages;
  VectorXi trainingLabels;
  VectorXi testImages;
  VectorXi testLabels;
}; 
void loadToStdVectors();
processedData loadToEigenMatrices();

#endif
