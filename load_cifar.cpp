
#include <iostream>
#include "cifar-10-master/include/cifar/cifar10_reader.hpp"   //Library that do the loading of the file
#include <Eigen/Dense> //The Eigen library for linear algebra
#include "load_cifar.h" //The header file

using namespace Eigen;

void loadToStdVectors(){
		//We load the dataset and store it in the object dataset
		auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

		//We extract training features and labels and test features and labels
		std::vector<uint8_t> trainingLabels = dataset.training_labels;
		std::vector<std::vector<uint8_t>> trainingImages = dataset.training_images;
		std::vector<uint8_t> testLabels = dataset.test_labels;
		std::vector<std::vector<uint8_t>> testImages = dataset.test_images;

		std::cout << "Size of training set: " << trainingLabels.size() << std::endl;
		std::cout << "Size of test set: " << testLabels.size() << std::endl;

		for(int i = 0; i < 10; i++){
		  std::cout << "train label " << i << ": " << unsigned(trainingLabels[i]) << std::endl;
		}

		for(int i = 0; i < 10; i++){
		  std::cout << "train image 0 pixel " << i << ": " << unsigned(trainingImages[0][i]) << std::endl;
		}
}

void loadToEigenMatrices(){
		//We load the dataset and store it in the object dataset
		auto dataset = cifar::read_dataset<VectorXi, VectorXi, int, int>();
		/*
		std::vector<uint8_t> trainingLabels = dataset.training_labels;
		vectorXd trainingLabelsEig(trainingLabels.size());
		
		std::vector<uint8_t> testLabels = dataset.test_labels;
		vectorXd testLabelsEig(testLabels.size());
		
		for(int i = 0; i < testLabels.size(); i++){
		  testLabelsEig(i) = testLabels[i];
		}
		*/
		
}


int main()
{
  loadToStdVectors();
  return 0;
}