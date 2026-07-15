// nccl_hello_solution.c  --  the ANSWER KEY (teacher copy).
// One process drives 4 GPUs on one node and does an all-reduce (sum).
// Every GPU starts with the value = its own rank. After all-reduce every GPU
// should hold 0+1+2+3 = 6.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

#define CUDA(x) do{ cudaError_t e=(x); if(e){printf("CUDA error %s: %s\n",#x,cudaGetErrorString(e)); return 1;} }while(0)
#define NCCLC(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){printf("NCCL error %s: %s\n",#x,ncclGetErrorString(r)); return 1;} }while(0)

int main(){
  const int nGPU = 4;
  const int N = 8;                 // 8 numbers per GPU
  int devs[4] = {0,1,2,3};

  float* sendbuff[4]; float* recvbuff[4]; cudaStream_t stream[4];
  for(int g=0; g<nGPU; g++){
    CUDA(cudaSetDevice(g));
    CUDA(cudaMalloc(&sendbuff[g], N*sizeof(float)));
    CUDA(cudaMalloc(&recvbuff[g], N*sizeof(float)));
    CUDA(cudaStreamCreate(&stream[g]));
    // fill this GPU's send buffer with the value = its rank g
    float host[N]; for(int i=0;i<N;i++) host[i]=(float)g;
    CUDA(cudaMemcpy(sendbuff[g], host, N*sizeof(float), cudaMemcpyHostToDevice));
  }

  // --- make one NCCL communicator per GPU (all in this one process) ---
  ncclComm_t comm[4];
  NCCLC(ncclCommInitAll(comm, nGPU, devs));

  // --- the all-reduce: every GPU adds its numbers, all get the sum ---
  // ncclGroupStart/End lets us submit all 4 GPUs together.
  NCCLC(ncclGroupStart());
  for(int g=0; g<nGPU; g++){
    NCCLC(ncclAllReduce(sendbuff[g], recvbuff[g], N, ncclFloat, ncclSum, comm[g], stream[g]));
  }
  NCCLC(ncclGroupEnd());

  // wait for all GPUs to finish
  for(int g=0; g<nGPU; g++){ CUDA(cudaSetDevice(g)); CUDA(cudaStreamSynchronize(stream[g])); }

  // check GPU 0: it should hold 0+1+2+3 = 6
  float out[N];
  CUDA(cudaSetDevice(0));
  CUDA(cudaMemcpy(out, recvbuff[0], N*sizeof(float), cudaMemcpyDeviceToHost));
  printf("GPU0 after all-reduce: %.0f %.0f %.0f ...  (expect 6)\n", out[0], out[1], out[2]);
  printf(out[0]==6.0f ? "RESULT OK\n" : "RESULT WRONG\n");

  for(int g=0; g<nGPU; g++){ ncclCommDestroy(comm[g]); cudaFree(sendbuff[g]); cudaFree(recvbuff[g]); }
  return out[0]==6.0f ? 0 : 1;
}
