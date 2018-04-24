# include <stdio.h>
# include <math.h>

// Global variables

const int rows = 2;
const int cols = 6;
const float data[6][6] = {{1, 3, 3, 5, 6, 7},{1, 2, 3, 43, 8, 99}};

float sigmoid(float x){
  return 1 / (1 + exp(-x));
}

float * pw_log(float array[]){
  float s = 0;
  static float data[cols];
  for (int i=0; i<cols; i++){
    data[i] = log(array[i]);
  }
  return data;
}

float * pw_exp(float array[]){
  float s = 0;
  static float data[cols];
  for (int i=0; i<cols; i++){
    data[i] = exp(array[i]);
  }
  return data;
}

float * pw_sigmoid(float array[]){
  float s = 0;
  static float data[cols];
  for (int i=0; i<cols; i++){
    data[i] = sigmoid(array[i]);
  }
  return data;
}

float dsigmoid(float x){
  float r;
  r = sigmoid(x);
  return  r * (1 - r);
}

float * pw_dsigmoid(float array[]){
  float s = 0;
  static float data[cols];
  for (int i=0; i<cols; i++){
    data[i] = dsigmoid(array[i]);
  }
  return data;
}

float sum(float array[], int size){
  float s = 0;
  for (int i=0; i<size; i++){
    s += array[i];
  }
  return(s);
}

float mean(float array[], int size){
  float s;
  s = sum(array, size);
  return s / size;
}

float * sum_on_rows(float array[rows][cols]){
  static float s[rows];
  float r;
  for (int j=0; j<rows; j++){
      s[j] = sum(array[j], cols);
    }
  return s;
}

float nll(float y_true, float y_pred){
  return 1;
}

int main(void){
  float array[3] = {1, 2, 3};
  float m;
  m = mean(array, 3);
  printf("%f\n",m);
  return 0;
}
