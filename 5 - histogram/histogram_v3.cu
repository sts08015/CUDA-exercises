%%writefile histogram1.cu

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

enum {FAIL,SUCCESS};

#define BINS 7
//DIV : number of elements in each bin
/*
  bin0  bin1  bin2  bin3  bin4  bin5  bin6
  4     4     4     4     4     4     2
*/
#define DIV ((26+BINS-1)/BINS)


/*
  IDEA: manage sub-histogram per thread block via shared memory and accumulate the results of sub-histogram to final histogram
*/
__global__ void histogram(char* arr, int* result, int n)
{
  __shared__ int shmem[BINS]; //shared memory for sub-histogram
  if(threadIdx.x < BINS) shmem[threadIdx.x] = 0;  //initialize shared mem
  __syncthreads();  //IMPORTANT!!

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int offset;
  for(int i=tid; i<n; i+=(gridDim.x * blockDim.x))  //number of total threads might be lesser than total elements
  {
    /*
      Since the number of thread blocks are divded by 4, more works are assigned to each threads. This bring about less atomicAdd operation towards device memory which leads to performance improvement.
    */
    offset = arr[i] - 'a';
    atomicAdd(&shmem[(offset/DIV)], 1); //to avoid race condition; multiple threads might be concurrently writing to same address
  }
  __syncthreads();

  if(threadIdx.x < BINS) atomicAdd(&result[threadIdx.x], shmem[threadIdx.x]);
}

void init(char* arr, int n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 25);

  for(int i=0;i<n;i++) arr[i] = 'a' + dis(gen);
}

int verifier(int *arr,int n)
{
  int cnt = 0;
  for(int i=0;i<BINS;i++) cnt += arr[i];
  if(cnt == n) return SUCCESS;
  else return FAIL;
}

int main()
{
  int n = 1<<20;

  char *h_a = new char[n];
  int *h_result = new int[BINS];
  memset(h_result,0,sizeof(int)*BINS);

  char *d_a;
  int *d_result;
  cudaMalloc(&d_a,n*sizeof(char));
  cudaMalloc(&d_result,BINS*sizeof(int));

  init(h_a,n);
  cudaMemcpy(d_a,h_a,n*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(d_result,h_result,BINS*sizeof(int),cudaMemcpyHostToDevice);

  int THREADS = 256;
  int BLOCKS = (int)ceil((float)n/(4*THREADS)); //This time, we divided the number of thread blocks by 4

  histogram<<<BLOCKS,THREADS>>>(d_a,d_result,n);

  cudaMemcpy(h_result,d_result,BINS*sizeof(int),cudaMemcpyDeviceToHost);

  int res = verifier(h_result,n);
  if(res == SUCCESS) std::cout << "YAY!" << std::endl;
  else std::cout << "Hmm.." << std::endl;

  delete[] h_a;
  delete[] h_result;
  cudaFree(d_a);
  cudaFree(d_result);

  return 0;
}