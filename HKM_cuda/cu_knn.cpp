#include <algorithm>
#include <iostream>
#include <time.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <assert.h>
#include "cu_knn.h"
#include "ccCommon.hpp"
#include "ccDistance.hpp"
using namespace std;


void free_mat(PDATA mat)
{
	free(mat->data);
	free(mat);
}

PDATA mat_product(PDATA mat1, PDATA mat2)
{
	if (mat1->_cols != mat2->_rows)
	{
		printf("this is not right\n");
		return NULL;
	}
	PDATA mat3 = new cData;
	mat3->data = (float *)malloc(sizeof(float)*(mat1->_rows)*(mat2->_cols));
	mat3->_rows = mat1->_rows;
	mat3->_cols = mat2->_cols;
	/*
	*INIT the matrix we want calculate
	* col primary
	*/
	{
		float *d_a, *d_b, *d_c;
		cublasInit();
		cublasAlloc((mat1->_cols)*(mat1->_rows), sizeof(float), (void **)&d_a);
		cublasAlloc((mat2->_cols)*(mat2->_rows), sizeof(float), (void **)&d_b);
		cublasAlloc((mat3->_rows)*(mat3->_cols), sizeof(float), (void **)&d_c);
		cudaMemcpy(d_a, mat1->data, sizeof(float)*(mat1->_cols)*(mat1->_rows), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, mat2->data, sizeof(float)*(mat2->_rows)*(mat2->_cols), cudaMemcpyHostToDevice);
		cublasHandle_t handle;
		cublasCreate(&handle);
		float alpha = 1.0;
		float beta = 0.0;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mat1->_rows, mat2->_cols,
			mat2->_rows, &alpha, d_a, mat1->_rows, d_b, mat2->_rows, &beta, d_c, mat1->_rows);
		cudaError_t cudaStatus=cudaMemcpy(mat3->data, d_c, sizeof(float)*(mat3->_rows)*(mat3->_cols), cudaMemcpyDeviceToHost);
		cublasShutdown();
		cublasFree(d_a);
		cublasFree(d_b);
		cublasFree(d_c);

	}
	/* need to trans the mat3*/
	return mat3;

}

void ele_mat_show(PDATA mat)
{
	int n;
	for (int i = 0; i<mat->_rows; i++){
		for (int j = 0; j<mat->_cols; j++){
			cout << mat->data[IDX2C(i, j, mat->_rows)] << "\t";
		}
		cout << endl;
	}
}
float myrand()
{
	return rand() % 10;
}
template<class Tret, class T>
void cuknn(Tret* dists, uint* ids, uint k, Data<T>& data1, Data<T>& data2)
{
	clock_t start, end;

#if 0
	for (int i = 0; i<M*N; i++)
	{
		cout << c[i] << "\t";
	}
	cout << endl;
#endif
	uint col1, col2, n1, n2, row, rows1, rows2;
	PDATA mat1, mat2, mat3;
	//-----
	fill(dists, dists + data1.size(), 0);                            
	/* remember to initialize the point*/
	mat1 = (PDATA)malloc(sizeof(cData));
	mat2 = (PDATA)malloc(sizeof(cData));
	mat3 = (PDATA)malloc(sizeof(cData));
	mat1->_rows = data2.getMaxDim();
	mat1->_cols = data2.size();
	//pair<T*, uint> point1 = data1.getPoint(col1), point2 = data2.getPoint(col2);
	/*for (row = 0; row < rows1; ++row)
	{
	t = (Tret)point1.first[row] - (Tret)point2.first[row];
	d += t * t;
	}*/
	pair<T*, uint> point1;
	int i,j,n;
	int c[10];//k=10 in this way
	mat1->data = (float *)malloc(sizeof(float)*mat1->_rows*mat1->_cols);
	
	for (int i = 0; i < mat1->_cols; i++)
	{
		point1 = data2.getPoint(i);
		c[i] = 0;
		for (j = 0; j < mat1->_rows; j++)
		{
			mat1->data[IDX2C(j, i, mat1->_rows)] = point1.first[j];
			c[i] = c[i] + (point1.first[j])*(point1.first[j]);
		}

			
			
	}
	//generate(mat1->data,mat1->data+(mat1->_cols)*(mat1->_rows),myrand);
	//ele_mat_show(mat1);
	mat2->_rows = data1.size();
	mat2->_cols = data1.getMaxDim();
	mat2->data = (float *)malloc(sizeof(float)*mat2->_rows*mat2->_cols);
	pair<T*, uint> point2;
	for (int i = 0; i < mat2->_rows; i++)
	{
		point2 = data1.getPoint(i);

		for (int j = 0; j < mat2->_cols; j++)
		{
			n = IDX2C(i, j, mat2->_rows);
			mat2->data[n] = point2.first[j];
		}

		//ele_mat_show(mat2);

	}
	//generate(mat2->data,mat2->data+(mat2->_cols)*(mat2->_rows),myrand);
	//ele_mat_show(mat2);
	mat3 = mat_product(mat2, mat1);

	int pc,m;
	for (int i = 0; i < mat3->_rows; ++i)
	{                                                         
		for (int j = 0; j < mat3->_cols; ++j)
		{
			pc = (mat3->data[IDX2C(i, j, mat2->_rows)]) * 2;
			m = pc - c[j];
			if (j == 0)
			{
				dists[i] = m;
				ids[i] = j;
			}
			else
			{
				if (m > dists[i])
				{
					dists[i] = m;
					ids[i] = j;
				}
			}

				
		}
		if (dists[j] < 0)
		{
			dists[j] = abs(dists[j]);
		}
 
	}                                                         
/*	for (int i = 0; i<mat3->_rows; i++)
	{
		for (int j = 0; j<mat3->_cols; j++)
		{
			cout << mat3->data[i + j*mat3->_rows] << '\t';
		}
		cout << endl;
	}*/
	free_mat(mat1);
	free_mat(mat2);
	free_mat(mat3);
}

#define CUKNN_F(T)      \
  template void cuknn(float*, uint*, uint, Data<T>&, Data<T>&);
#define CUKNN_D(T)      \
  template void cuknn(double*, uint*, uint, Data<T>&, Data<T>&);

TEMPLATE(CUKNN_F)
TEMPLATE(CUKNN_D)