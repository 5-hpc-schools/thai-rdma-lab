// nccl_hello.c  --  YOUR TASK.  Fill in the 2 lines marked TODO, then build and run.
// One process drives 4 GPUs on one node and does an all-reduce (sum).
// Every GPU starts with the value = its own rank. After all-reduce every GPU
// should hold 0+1+2+3 = 6.  If your code is right, you will see "RESULT OK".
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

#define CUDA(x) do{ cudaError_t e=(x); if(e){printf("CUDA error %s: %s\n",#x,cudaGetErrorString(e)); return 1;} }while(0)
#define NCCLC(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){printf("NCCL error %s: %s\n",#x,ncclGetErrorString(r)); return 1;} }while(0)

int main(){
  const int nGPU = 4;
  const int N = 8;
  int devs[4] = {0,1,2,3};

  float* sendbuff[4]; float* recvbuff[4]; cudaStream_t stream[4];
  for(int g=0; g<nGPU; g++){
    CUDA(cudaSetDevice(g));
    CUDA(cudaMalloc(&sendbuff[g], N*sizeof(float)));
    CUDA(cudaMalloc(&recvbuff[g], N*sizeof(float)));
    CUDA(cudaStreamCreate(&stream[g]));
    float host[N]; for(int i=0;i<N;i++) host[i]=(float)g;   // this GPU's value = its rank
    CUDA(cudaMemcpy(sendbuff[g], host, N*sizeof(float), cudaMemcpyHostToDevice));
  }

  ncclComm_t comm[4];
  // TODO 1: make one communicator per GPU, all in this process.
  //   Write exactly:   NCCLC(ncclCommInitAll(comm, nGPU, devs));
  //   (comm = the array to fill, nGPU = how many GPUs, devs = which GPU ids)


  NCCLC(ncclGroupStart());
  for(int g=0; g<nGPU; g++){
    // TODO 2: all-reduce (SUM) N floats from sendbuff[g] into recvbuff[g] on comm[g], stream[g].
    //   Write exactly:
    //   NCCLC(ncclAllReduce(sendbuff[g], recvbuff[g], N, ncclFloat, ncclSum, comm[g], stream[g]));

  }
  NCCLC(ncclGroupEnd());

  for(int g=0; g<nGPU; g++){ CUDA(cudaSetDevice(g)); CUDA(cudaStreamSynchronize(stream[g])); }

  float out[N];
  CUDA(cudaSetDevice(0));
  CUDA(cudaMemcpy(out, recvbuff[0], N*sizeof(float), cudaMemcpyDeviceToHost));
  printf("GPU0 after all-reduce: %.0f %.0f %.0f ...  (expect 6)\n", out[0], out[1], out[2]);
  printf(out[0]==6.0f ? "RESULT OK\n" : "RESULT WRONG\n");

  for(int g=0; g<nGPU; g++){ ncclCommDestroy(comm[g]); cudaFree(sendbuff[g]); cudaFree(recvbuff[g]); }
  return out[0]==6.0f ? 0 : 1;
}
