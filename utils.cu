#include <iostream>
#include <random>


float grad_i(float theta, int i){
  grad = 2 * (y_true - y_pred);
  return grad;
}

float sgd(float theta, float size, int nitr){
  int i;
  for (int i=0; i<nitr; i++){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, size);
    i = dis(gen);
    theta = theta - stepsize * grad_i(theta, i);
  }
  return theta;
}


int main()

{
  float theta={1, 2, 3};
  float size = 3;
  int nitr = 10;




return 0;

}
