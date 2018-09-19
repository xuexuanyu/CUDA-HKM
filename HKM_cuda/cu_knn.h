#include <algorithm>
#include <iostream>
#include <time.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <assert.h>
#include "ccCommon.hpp"
#include "ccDistance.hpp"

#define  IDX2C(i,j,leading) (((j)*(leading))+(i))


typedef struct _data *PDATA;
typedef struct _data
{
	int _rows;
	int _cols;
	float *data;
} cData;


void free_mat(PDATA mat);
PDATA mat_product(PDATA mat1, PDATA mat2);
void ele_mat_show(PDATA mat);
template<class Tret, class T>
void cuknn(Tret* dists, uint* ids, uint k, Data<T>& data1, Data<T>& data2);

/**/