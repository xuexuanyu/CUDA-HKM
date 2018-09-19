#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <helper_cuda.h>
#include <helper_functions.h>
#include "average.h"
#include <stdio.h>
#include "ccCommon.hpp"
#include "ccDistance.hpp"


__global__ void getblockid_Kernel(uint* arr_div, uint* ids, uint size, uint* nid_div)
{
	int i = threadIdx.x;
	uint m=i*(size/5);
	uint sum=0;
	int j;
	for (j = 0; j < size; ++j)
	{
		if (ids[j] == i)
		{
			arr_div[m] = j;
			sum++;
			m++;

		}
	}
	nid_div[i] = sum;

}

__global__ void getmean_Kernel(uint *arr_div, uint *data, uint *mean_div, uint *nid_div)
{
	int k = blockIdx.x;
	int dim = threadIdx.x;
	int size;
	size=nid_div[k];
	uint tmean=0;
	uint m = 0;
	for (int i = 0; i < k-1; ++i)
	{
		m += nid_div[i-1];
	}
	for(int i = 0; i < size; ++i)
	{
		uint id = arr_div[m+i];
		tmean += data[id * 64 + dim];
	}
	mean_div[k*64+dim] = tmean;
	
}

template <typename T>
cudaError_t getidWithCuda(Data<T>& data1, uint* ids, uint* nid_host, uint *d_host)
{

	cudaError_t cudaStatus;
	uint nids = data1.size();
	uint **addr_div;
	uint *nid_div;
	uint *ids_div;
	uint *arr_div;
	cudaStatus = cudaMalloc((void**)&arr_div, nids * 2 * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&ids_div, nids*sizeof(uint));
	cudaStatus = cudaMalloc((void**)&nid_div, sizeof(uint) * 10);
	cudaStatus = cudaMemcpy(ids_div, ids, nids*sizeof(uint), cudaMemcpyHostToDevice);
	getblockid_Kernel << <1, 10 >> >(arr_div,ids_div, nids, nid_div);
	cudaStatus = cudaMemcpy(nid_host, nid_div, sizeof(uint)*10, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(d_host, arr_div, nids * 2 * sizeof(uint), cudaMemcpyDeviceToHost);
	int sum=0;
	for (int i = 0; i < 10; ++i)
	{
		sum += nid_host[i];
	}
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaFree(arr_div);
	cudaFree(ids_div);
	cudaFree(nid_div);
	//remalloc
	return cudaStatus;
}




template <typename T>
void getmeanbycuda(Data<T>& data1, uint* ids, uint *meansbycuda)
{
	cudaError_t cudaStatus;
	//cudaStatus = cudaDeviceReset();

	uint *nid_host=new uint[10];
	uint size = data1.size();
	uint *ids_arr = new uint[size*2];
	cudaStatus=getidWithCuda(data1, ids, nid_host, ids_arr);//get ids_arr
	//-----re-arr--------------
	uint *idarray = new uint[size];
	uint m = 0;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < nid_host[i]; ++j)
		{
			idarray[m] = ids_arr[i*(size / 5) + j];
			m++;
		}
	}
	//------get need data-----------------
	uint nbytes = data1.size() * data1.ndims;
	uint *data_host = new uint[nbytes];
	for (int i = 0; i < data1.size(); ++i)
	{
		uint pid = data1.filterId(i);
		for (int j = 0; j < data1.ndims; ++j)
		{
			data_host[i*data1.ndims + j] = data1.data.fixed.data.full[pid * data1.ndims + j];
		}

	}
	//-----------------
	uint *data2;
	uint *mean_div;
	uint *nid_div;
	uint *arr_div;
	uint *mean_host = new uint[64 * 10];
	cudaStatus = cudaMalloc((void**)&nid_div, sizeof(uint) * 10);
	cudaStatus = cudaMalloc((void**)&data2, nbytes*sizeof(uint));
	cudaStatus = cudaMalloc((void**)&mean_div, 10 * 64 * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&arr_div, size*sizeof(uint));
	cudaStatus = cudaMemcpy(arr_div, idarray, size*sizeof(uint), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(data2, data_host, nbytes*sizeof(uint), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(nid_div, nid_host, sizeof(uint) * 10, cudaMemcpyHostToDevice);
	
	getmean_Kernel << <10, 64 >> >(arr_div, data2, mean_div, nid_div);
	//cudaStatus = cudaMemcpy(mean_host, mean_div, 10 * 64 * sizeof(uint), cudaMemcpyDeviceToHost);
	/*for (int i = 0; i < 10; ++i)
	{
		for (int j = 0; j < 64; ++j)
		{
			meansbycuda[i * 64 + j] = (mean_host[i * 64 + j]) / nid_host[i];
		}
	}*/
	
	
	delete[] data_host;
	delete[] ids_arr;
	delete[] idarray;
	delete[] nid_host;
	
}
/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, size_t size)
{
	int *dev_a = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, size, size * sizeof(int), 0 >> >(dev_c, dev_a);
		// cudaThreadSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	return cudaStatus;
}*/


#define GETMEAN_F(T)      \
  template void getmeanbycuda(Data<T>& ,uint*,uint *);
#define GETMEAN_D(T)      \
  template void getmeanbycuda(Data<T>& ,uint*,uint *);

TEMPLATE(GETMEAN_F)
TEMPLATE(GETMEAN_D)