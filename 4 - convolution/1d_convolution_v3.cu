#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>

enum {FAIL, SUCCESS};

#define MASK_LENGTH 7
__constant__ int mask[MASK_LENGTH]; //constant memory --> mask doesnt change throughout the entire loop!

/*
  IDEA: There are many common elements used to calculate convolution value between consecutive threads. So, let's utilize shared memory.
  Opt to not use additional thread just for loading padded value. --> Some threads have to load two elements.
*/
__global__ void convolution_1d(int* arr, int* res)
{
  extern __shared__ int shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int r = MASK_LENGTH/2;
  int n_padded = blockDim.x + r*2;  //size of padded shared memory
  int offset = threadIdx.x + blockDim.x;  //offset for the second set of loads in shared mem
  int g_offset = blockDim.x * blockIdx.x + offset;  //global offset for the array in device memory

  shmem[threadIdx.x] = arr[tid];
  if(offset < n_padded) shmem[offset] = arr[g_offset];  //load second element if meets this condition.
  __syncthreads();

  int tmp = 0;
  for(int i=0;i<MASK_LENGTH;i++) tmp += shmem[threadIdx.x+i] * mask[i];
  res[tid] = tmp;
}

int verifier(int *arr,int *mask, int *res,int n,int m)
{
  //Note that arr is padded here
  for(int i=0;i<n;i++)
  {
    int tmp = 0;
    for(int j=0;j<m;j++)
    {
      tmp += arr[i+j] * mask[j];
    }
    if(tmp != res[i]) return FAIL;
  }

  return SUCCESS;
}

int main(void)
{
  int n = 1 << 20;
  int size_n = sizeof(int) * n;

  int m = MASK_LENGTH;  //mask size
  int size_m = sizeof(int) * m;

  int r = MASK_LENGTH/2;
  int n_p = n + r*2;  //padded length: [padding] + arr + [padding]
  int size_n_p = n_p * sizeof(int);

  int *h_arr = new int[n_p];
  int *h_mask = new int[m];
  int *h_result = new int[n];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 99);

  for(int i=0;i<n_p;i++)
  {
    if(i<r || i>=n+r) h_arr[i] = 0; //zero padding
    else h_arr[i] = dis(gen);
  }
  for(int i=0;i<m;i++) h_mask[i] = dis(gen);

  int *d_arr, *d_result;
  cudaMalloc(&d_arr,size_n_p);
  cudaMalloc(&d_result,size_n);

  cudaMemcpy(d_arr,h_arr,size_n_p,cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(mask,h_mask,size_m);

  int THREADS = 256;
  int GRID = (int)ceil((float)n/THREADS);
  size_t SHMEM = (THREADS + r*2) * sizeof(int); //dynamically allocate shared memory

  convolution_1d <<< GRID, THREADS, SHMEM>>>(d_arr,d_result);

  cudaMemcpy(h_result,d_result,size_n,cudaMemcpyDeviceToHost);
  int val = verifier(h_arr,h_mask,h_result,n,m);

  if(val == SUCCESS) std::cout << "YAY!!" << std::endl;
  else std::cout << "Hmm..." << std::endl;

  delete[] h_arr;
  delete[] h_mask;
  delete[] h_result;
  cudaFree(d_arr);
  cudaFree(d_result);
  return 0;
}