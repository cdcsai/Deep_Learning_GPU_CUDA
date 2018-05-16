#include <iostream>
#include </usr/users/hpcgif/hpcgif_9/Deep_Learning_GPU_CUDA/Eigen/Dense>
//#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <random>

using namespace Eigen;
using namespace std;

// CUDA functions used for training the NN
__global__ void dot_par(int*ret, int *w, int *X){
  ret[threadIdx.x] = w[threadIdx.x] * X[threadIdx.x];
  __syncthreads();
  if (0 == threadIdx.x){
    int sum = 0;
    for (int i=0; i<26; i++)
      sum += ret[i];
    ret* = sum;

  }}

int main(){

      int *ret, int *w, int *X;
    	//VectorXf w, dw;
    	//float b, db, cost;
    	//initialize(w, b, 4);
    	// MatrixXf x(4, 26);
    	// x << 1, 0, 3, 4, 1, 2, 2, 3, 4, 2, 3, 5, 1, 2, 3, 2, 5, 0, 1, 4, 5, 0, 1, 2, 1, 3,
      //      2, 5, 3, 5, 2, 1, 4, 3, 2, 0, 0, 2, 4, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1, 2, 5,
    	//    2, 3, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1, 2, 5, 3, 5, 2, 1, 4, 3, 2, 0, 0, 2, 4,
    	//    2, 3, 2, 5, 0, 1, 4, 5, 0, 1, 2, 1, 3, 2, 3, 0, 5, 3, 2, 4, 2, 1, 1, 2, 2, 1;
    	// MatrixXf xTest(4, 5);
    	// xTest << 1, 3, 4, 5, 3,
    	//          0, 4, 5, 4, 4,
    	// 		 2, 3, 2, 1, 1,
    	// 		 3, 4, 0, 0, 5;
    	// RowVectorXf y(26), yTest(5);
    	// yTest << 1, 1, 1, 0, 0;
    	// RowVectorXf yPredictions, yPredictionsTest;
    	// y << 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0;
    	//vector<float> costs;

      dot_par<<<1, 1>>>(ret, w, X);

    	//propagate(VectorXf w, float b, MatrixXf X, RowVectorXf y, VectorXf &dw, float &db, float &cost);

    	//propagate_i(w, b, x.col(3), y(3), dw, db, cost_i);
    	//model(x, y, xTest, yTest, yPredictions, yPredictionsTest, w, b, costs, 10000, 0.0002);

    	return(0);
    }
