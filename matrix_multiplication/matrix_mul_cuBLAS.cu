//nvcc -o matrix matrix.cu -lcublas -lcurand
#include <iostream>
#include <cmath>
#include <cublas_v2.h>
#include <curand.h>

enum {FAIL, SUCCESS};

int verifier(const float* mat1, const float* mat2, const float* res, int n)
{
  //cuBLAS --> matrix is assumed as column major order
    float epsilon = 0.001;
    for(int i=0;i<n;i++)
    {
      for(int j=0;j<n;j++)
      {
        int val = 0;
        for(int k=0;k<n;k++)
        {
          val += mat1[n*k + i] * mat2[n*j + k]; //indexing changes!
        }
        if(fabs(res[n*j + i] - val) >= epsilon) return FAIL;
      }
    }
    return SUCCESS;
}

int main(void)
{
  int n = 1<<10; // 1024 x 1024 matrix
  int element_num = n*n;
  size_t size = element_num*sizeof(float);
  
  float *h_a, *h_b, *h_c;
  h_a = new float[element_num];
  h_b = new float[element_num];
  h_c = new float[element_num];

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a,size);
  cudaMalloc(&d_b,size);
  cudaMalloc(&d_c,size);

  //Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  //Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  //Initialize the matrices on the device
  curandGenerateUniform(prng,d_a,element_num);
  curandGenerateUniform(prng,d_b,element_num);

  //cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;
  
  //c = (alpha*a) * b + (beta*c)
  //mxn * n*k = m*k
  //lda stands for leading dimension of A
  //cublasSgemm(handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

  cudaMemcpy(h_a,d_a,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);
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