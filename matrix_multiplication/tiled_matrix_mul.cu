//Mystery: how come the performance of the tiling algorithm is worse than the basic one? hmm....
//maybe matrix size isn't big enough

#include <iostream>
#include <random>
#include <cmath>

enum {FAIL, SUCCESS};

#define SM_LEN 1024
//32*32

__global__ void tiled_matrix_mul(const int* mat1, const int* mat2, int* res, const int n)
{
  __shared__ int SM1[SM_LEN]; //shared with all threads in a block. CTA-private.
  __shared__ int SM2[SM_LEN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  int tile_size = blockDim.x;

  int tmp = 0;
  if(row < n && col < n)
  {
    int sm_idx = ty * tile_size + tx;
    for(int i=0;i<n;i+=tile_size)
    {
      SM1[sm_idx] = mat1[i + tx + row*n];
      SM2[sm_idx] = mat2[(i + ty)*n + col];
      __syncthreads();  //to assure SM1 and SM2 for a corresponding thread block are completely filled

      for(int j=0;j<tile_size;j++)
      {
        tmp += SM1[ty*tile_size + j] * SM2[j*tile_size + tx];
      }
      __syncthreads();  //to assure multiplying elements in SM is done before loading next values from device memory
    }

    res[row*n + col] = tmp;
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
  
  const int BLOCK_SIZE = (int)sqrt(props.maxThreadsPerBlock); //32
  const int GRID_SIZE = (int)ceil(n/BLOCK_SIZE);  //32

  dim3 grid(GRID_SIZE,GRID_SIZE);
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

  tiled_matrix_mul <<<grid, threads>>>(d_a,d_b,d_c,n);

  cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

  if(verifier(h_a,h_b,h_c,n) == SUCCESS) std::cout<< "YAY!" << std::endl;
  else std::cout << "Hmm.." << std::endl;

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}