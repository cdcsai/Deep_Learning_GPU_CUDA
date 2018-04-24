# include <stdio.h>


__global__ void cube(float *d_in, float *d_out){
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f * f;
}

int main(){
  float *h_in, *h_out;
  const int ARRAYSIZE = 64;
  const int ARRAYBYTE = 64 * sizeof(float);

  // Generate input ARRAY

  h_in = (float*)malloc(ARRAYBYTE);
	h_out = (float*)malloc(ARRAYBYTE);

  for (int i = 0; i<ARRAYSIZE; i++){
    h_in[i] = float(i);
  }


  // Declare GPU memory pointer

  float * d_in;
  float * d_out;

  cudaMalloc((void **) &d_in, ARRAYBYTE);
  cudaMalloc((void **) &d_out, ARRAYBYTE);

  cudaMemcpy(d_in, h_in, ARRAYBYTE, cudaMemcpyHostToDevice);
  cube<<<1, ARRAYSIZE>>>(d_out, d_in);
  cudaMemcpy(h_out, d_out, ARRAYBYTE, cudaMemcpyDeviceToHost);

  for (int i = 0; i<ARRAYSIZE; i++){
    printf("%f\n", h_out[i]);
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
  }
}
