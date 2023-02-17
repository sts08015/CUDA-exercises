#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>

enum {FAIL, SUCCESS};

// 7x7 mask
#define MASK_DIM 7
#define MASK_OFFSET (MASK_DIM/2)

__constant__ int mask[MASK_DIM * MASK_DIM];

__global__ void convolution_2d(int* matrix, int* res, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  int tmp = 0;
  for(int i=0;i<MASK_DIM;i++)
  {
    for(int j=0;j<MASK_DIM;j++)
    {
      if(start_r+i<0 || start_c+j<0 || start_r+i>=n || start_c+j>=0) continue;
      tmp += mask[i*MASK_DIM + j] * matrix[(start_r+i)*n + start_c+j];
    }
  }
  res[row*n + col] = tmp;
}

int verifier(int *matrix,int *mask, int *res,int n)
{
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<n;j++)
    {
      int tmp = 0;
      int start_r = i - MASK_OFFSET;
      int start_c = j - MASK_OFFSET;

      for(int a=0;a<MASK_DIM;a++)
      {
        for(int b=0;b<MASK_DIM;b++)
        {
          if(start_r+a<0 || start_c+b<0 || start_r+a>=n || start_c+b>=0) continue;
          tmp += mask[a*MASK_DIM + b] * matrix[(start_r+a)*n + start_c];
        }
      }

      if(tmp != res[i*n + j]) return FAIL;
    }
  }

  return SUCCESS;
}

void init(int* arr, int n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 99);

  for(int i=0;i<n;i++)
  {
    for(int j=0;j<n;j++)
    {
      arr[i*n + j] = dis(gen);
    }
  }
}

int main(void)
{
  // 1024 x 1024 matrix
  int n = 1 << 10;
  size_t size_n = sizeof(int) * n * n;
  size_t size_m = sizeof(int) * MASK_DIM * MASK_DIM;
  
  int* matrix = new int[n*n];
  int* result = new int[n*n];
  int* h_mask = new int[MASK_DIM*MASK_DIM];

  init(matrix,n);
  init(h_mask,MASK_DIM);

  int *d_matrix;
  int *d_result;
  cudaMalloc(&d_matrix,size_n);
  cudaMalloc(&d_result,size_n);


  cudaMemcpy(d_matrix,matrix,size_n,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask,h_mask,size_m);

  int THREADS = 16;
  int BLOCKS = (int)ceil((float)n/THREADS);

  dim3 block_dim(THREADS,THREADS);
  dim3 grid_dim(BLOCKS,BLOCKS); //hmm?

  convolution_2d <<<grid_dim,block_dim>>>(d_matrix,d_result,n);

  cudaMemcpy(result,d_result,size_n,cudaMemcpyDeviceToHost);

  int val = verifier(matrix,h_mask,result,n);
  if(val == SUCCESS) std::cout << "YAY!!" << std::endl;
  else std::cout << "Hmm.." << std::endl;

  delete[] matrix;
  delete[] result;
  delete[] h_mask;
  cudaFree(d_matrix);
  cudaFree(d_result);
  
  return 0;
}