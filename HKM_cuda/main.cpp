#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "ccHKmeans.hpp"
//#include "average.h"
//#include "mxData.hpp"
#define DIM 128


template <typename T>
void  fillData(Data<T>& data, T* points, long size, bool copy = false)
{
	uint npoints = size;
	{
		data.type = DATA_FIXED;
		data.npoints = size / DIM;
		data.ndims = DIM;
		fillMatrix(data.data.fixed, points, size, copy);

	}

}
template<class T>
void fillMatrix(Matrix<T>& mat, T* point, long size, bool cp = false)
{
	//clear
	mat.clear();

	//get rows and columns
	size_t m = DIM;
	size_t n = size / DIM;
	{
		mat.set(point, m, n, cp);
	}
}
template <typename T>
void hkm(uint* buffer, long size)
{
	Data<T> data;
	HkmOptions opt;
	opt.dist = DISTANCE_L2;
	opt.nbranches = 10;
	opt.nlevels = 3;
	opt.nchecks = 1;
	opt.ntrees = 1;
	opt.usekdt = false;
	opt.niters = 20;
	Hkms<T>* hkm = new Hkms<T>(opt);
	fillData(data, (T*)buffer, size, false);

	hkm->create(data);
	data.clear();
}

void main()
{
	//--------------------
	//--------------------
	uint size = 10000;
	uint *point = new uint[size*DIM];
	for (int i = 0; i < size*DIM; ++i)
	{
		point[i] = 1 + (uint)(255 * rand() / (RAND_MAX + 1.0));
	}

	hkm<uint>(point, size*DIM);

	//fillData(data, dataIn, false);

	//---------------
	//Hkms<T>* hkm = new Hkms(opt);
	//hkm->create(data);
	//data.clear();


}
