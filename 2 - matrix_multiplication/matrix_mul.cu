#include <iostream>
#include <random>
#include <cmath>

enum {FAIL, SUCCESS};

__global__ void matrix_mul(const int* mat1, const int* mat2, int* res, const int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < n && col < n)
  {
    int sum = 0;	//utilizing variable sum instead of directly initializing res[row*n+col] significantly improves the performance (up to twice! according to my google colab environment)
    for(int k=0;k<n;k++)
    {
      sum += mat1[row*n + k] * mat2[k*n + col]; 
    }
    res[row*n + col] = sum;
  }
}

void init_matrix(int* mat, const int len)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 99);
    
    for(int i=0;i<len;i++) mat[i] = dis(gen);
}

int verifier(int* mat1, int* mat2, int* res, int n)
{
    for(int i=0;i<n;i++)
    {
      for(int j=0;j<n;j++)
      {
        int val = 0;
        for(int k=0;k<n;k++)
        {
          val += mat1[n*i + k] * mat2[n*k + j];
        }
        if(res[n*i + j] != val) return FAIL;
      }
    }
    return SUCCESS;
}

int main(void)
{
  int n = 1<<10; // 1024 x 1024 matrix
  int element_num = n*n;
  size_t size = element_num*sizeof(int);
  
  int *h_a, *h_b, *h_c;
  h_a = new int[element_num];
  h_b = new int[element_num];
  h_c = new int[element_num];

  init_matrix(h_a,element_num);
  init_matrix(h_b,element_num);

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a,size);
  cudaMalloc(&d_b,size);
  cudaMalloc(&d_c,size);

  cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);
  
  int devId;
  cudaGetDevice(&devId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, devId);
  
  const int BLOCK_SIZE = (int)sqrt(props.maxThreadsPerBlock);
  const int GRID_SIZE = (int)ceil((float)n/BLOCK_SIZE);

  dim3 grid(GRID_SIZE,GRID_SIZE);
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

  matrix_mul <<<grid, threads>>>(d_a,d_b,d_c,n);

  cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

  if(verifier(h_a,h_b,h_c,n) == SUCCESS) std::cout<<"YAY!" << std::endl;
  else std::cout << "Hmm.." << std::endl;

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}