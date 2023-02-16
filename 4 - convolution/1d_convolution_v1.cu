#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>

enum {FAIL, SUCCESS};

int verifier(int *arr,int *mask, int *res,int n,int m)
{
  int rad = m/2;
  for(int i=0;i<n;i++)
  {
    int tmp = 0;
    int s = i-rad;
    for(int j=0;j<m;j++)
    {
      if(s+j >=0 && s+j < n) tmp += arr[s+j] * mask[j];
    }
    if(tmp != res[i]) return FAIL;
  }

  return SUCCESS;
}

void init_vector(int* vec,const int len)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 99);
    
    for(int i=0;i<len;i++) vec[i] = dis(gen);
}

__global__ void convolution_1d(int* arr, int* mask, int* res, int n, int m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid>=n) return;

  int rad = m/2;
  int s = tid - rad;

  int tmp = 0;
  for(int i=0;i<m;i++)
  {
    if(s+i >=0 && s+i < n) tmp += arr[s+i] * mask[i];
  }
  res[tid] = tmp;
}

int main(void)
{
  int n = 1 << 20;
  int size_n = sizeof(int) * n;

  int m = 7;  //mask size
  int size_m = sizeof(int) * m;

  int *h_arr = new int[n];
  int *h_mask = new int[m];
  int *h_result = new int[n];

  init_vector(h_arr,n);
  init_vector(h_mask,m);

  int *d_array, *d_mask, *d_result;

  cudaMalloc(&d_array,size_n);
  cudaMalloc(&d_mask,size_m);
  cudaMalloc(&d_result,size_n);

  cudaMemcpy(d_array,h_arr,size_n,cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask,h_mask,size_m,cudaMemcpyHostToDevice);

  int THREADS = 256;
  int GRID = (int)ceil((float)n/THREADS);

  convolution_1d <<< GRID, THREADS>>>(d_array,d_mask,d_result,n,m);

  cudaMemcpy(h_result,d_result,size_n,cudaMemcpyDeviceToHost);
  int val = verifier(h_arr,h_mask,h_result,n,m);

  if(val == SUCCESS) std::cout << "YAY!!" << std::endl;
  else std::cout << "Hmm..." << std::endl;

  delete[] h_arr;
  delete[] h_mask;
  delete[] h_result;
  cudaFree(d_array);
  cudaFree(d_mask);
  cudaFree(d_result);
  return 0;
}