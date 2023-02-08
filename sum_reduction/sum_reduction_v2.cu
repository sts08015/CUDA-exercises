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
  /*
    Unlike previous code, only utilizing even indexed threads, we utilize consecutive threads here.

    We resolved warp divergence problem and removed costly division operation.

    Q. Why is it important to resolve warp divergence problem?
    A. Imagine the case when the length of a vector is long enough to set the stride value to 32, which is the warp size.
    In that case, in order to accumulate a partial sum, we need multiple warps and within each warp, only one thread's active mask is set which can significantly degrade the performance.
    However, if we use sequential threads to perform the same task, we can calculate the sum while fully utilizing the threads within a warp.

    Remaining Problem : Bank Conflict
  */

  __shared__ int partial_sum[SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  for(int stride=1; stride< blockDim.x; stride*= 2)
  {
    int idx = threadIdx.x * 2 * stride;
    if(idx + stride < blockDim.x) partial_sum[idx] += partial_sum[idx + stride];
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
  int GRID_SIZE = (int)ceil((float)n/TB_SIZE);

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