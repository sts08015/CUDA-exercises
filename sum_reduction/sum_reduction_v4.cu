#include <iostream>
#include <cstdlib>
#include <cmath>

//TB_SIZE
#define SIZE 256

enum {FAIL, SUCCESS};

void init(int* vec,const int len)
{
  for(int i=0;i<len;i++) vec[i] = 1;
}

__global__ void sum_reduction(int* v, int* result)
{
  __shared__ int partial_sum[SIZE];

  //IDEA: Halve the number of blocks, and replace single load with two loads and first add of the reduction
  //Why halve the number of blocks, not threads/block? --> each block is allocated to SM and we want to fully utilize each SM
  // Load elements and do first add of reduction
  int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  partial_sum[threadIdx.x] = v[tid] + v[tid + blockDim.x];
  __syncthreads();

  for(int stride = blockDim.x>>1 ; stride > 0; stride>>=1)
  {
    if(threadIdx.x < stride) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
    __syncthreads();
  }

  if(threadIdx.x == 0) result[blockIdx.x] = partial_sum[0];
}

int main(void)
{
  int n = 1<<16;
  size_t size = n * sizeof(int);

  int *h_v, *h_result;
  int *d_v, *d_result;

  h_v = new int[n];
  h_result = new int[n];
  cudaMalloc(&d_v,size);
  cudaMalloc(&d_result,size);

  //Initialize vector
  init(h_v, n);

  cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);

  int TB_SIZE = SIZE;
  int GRID_SIZE = (int)ceil((float)n/(2*TB_SIZE));

  sum_reduction <<<GRID_SIZE, TB_SIZE>>> (d_v, d_result);
  sum_reduction <<<1, GRID_SIZE>>> (d_result, d_result);

  cudaMemcpy(h_result,d_result,size, cudaMemcpyDeviceToHost);

  std::cout << "Accumulated result : " << h_result[0] << std::endl;
  
  delete[] h_v;
  delete[] h_result;
  cudaFree(d_v);
  cudaFree(d_result);

  return 0;
}