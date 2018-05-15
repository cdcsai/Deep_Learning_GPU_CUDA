
#include <iostream>
#include <stdio.h>
#include "cifar-10-master/include/cifar/cifar10_reader.hpp"   //Library that do the loading of the file
#include <Eigen/Dense> //The Eigen library for linear algebra
#include "load_cifar.h" //The header file

using namespace Eigen;

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}

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

processedData loadToEigenMatrices(){
		//We load the dataset and store it in the object dataset
		auto dataset = cifar::read_dataset<std::vector, std::vector, int, int>();
		//We flatten the training image container and we transform it to an Eigen matrix
		int nbTrainingImages = dataset.training_images.size();
		int sizeOfImage = dataset.training_images[0].size();
		std::vector<int> trainingImages = flatten(dataset.training_images);
		VectorXi trainingImagesEig = Map<VectorXi, Unaligned>(trainingImages.data(), trainingImages.size());
		MatrixXi trainingImagesMat = trainingImagesEig;
		trainingImagesMat.resize(sizeOfImage, nbTrainingImages);
		std::cout << trainingImagesMat.rows() << " " << trainingImagesMat.cols() << std::endl;
		//We flatten the test image container and we transform it to an Eigen vector
		std::vector<int> testImages = flatten(dataset.test_images);
		VectorXi testImagesEig = Map<VectorXi, Unaligned>(testImages.data(), testImages.size());
		//We transform the training labels container to an Eigen vector
		std::vector<int> trainingLabels = dataset.training_labels;
		VectorXi trainingLabelsEig = Map<VectorXi, Unaligned>(trainingLabels.data(), trainingLabels.size());
		//We transform the test labels container to an Eigen vector
		std::vector<int> testLabels = dataset.test_labels;
		VectorXi testLabelsEig = Map<VectorXi, Unaligned>(testLabels.data(), testLabels.size());
		
		processedData data;
		data.trainingImages = trainingImagesEig;
		data.trainingLabels = trainingLabelsEig;
		data.testImages = testImagesEig;
		data.testLabels = testLabelsEig;
		
		std::cout << trainingImagesMat(1, 0) << std::endl;
		std::cout << trainingImagesEig(50000) << std::endl;
		std::cout << "Nb training images: " << trainingImagesEig.size() << std::endl;
		std::cout << "Nb training labels: " << trainingLabels.size() << std::endl;
		std::cout << "Nb test images: " << testImagesEig.size() << std::endl;
		std::cout << "Nb test labels: " << testLabelsEig.size() << std::endl;
		
		return(data);
}


int main()
{
  processedData data = loadToEigenMatrices();
  return 0;
}