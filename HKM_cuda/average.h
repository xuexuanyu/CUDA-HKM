#include <vector>
#include <deque>
#include <queue>
#include <list>
#include <climits>
#include <cmath>

#include "ccCommon.hpp"
#include "ccHKmeans.hpp"
template <typename T>
void getmeanbycuda(Data<T>& data1, uint* ids, uint *meansbycuda);