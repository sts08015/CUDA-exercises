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

__device__ void warpReduce(volatile int* shmem, int t)
{
  /*
    __syncthreads is not necessary here!
    Reason: only one warp is active here. __syncthreads is for synchronizing multiple warps within a thread block.
    However, we stil need "volatile" keyword to ensure data is written to shared memory.
  */
  shmem[t] += shmem[t + 32];
  shmem[t] += shmem[t + 16];
  shmem[t] += shmem[t + 8];
  shmem[t] += shmem[t + 4];
  shmem[t] += shmem[t + 2];
  shmem[t] += shmem[t + 1];
}

__global__ void sum_reduction(int* v, int* result)
{
  __shared__ int partial_sum[SIZE];

  /*
    We focused on many warps that just do nothing in a loop this time.
    Terminating the loop when the stride becomes 32 allows a lot of unnecessary warps from being scheduled.
    Thus, we can improve performance.
  */
  int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  partial_sum[threadIdx.x] = v[tid] + v[tid + blockDim.x];
  __syncthreads();

  int stride;
  for(stride = blockDim.x>>1 ; stride > 32; stride>>=1) //1warp = 32threads
  {
    if(threadIdx.x < stride) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
    __syncthreads();
  }

  if(threadIdx.x < stride) warpReduce(partial_sum,threadIdx.x);

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