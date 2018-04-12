#ifndef OCTREE_H
#define OCTREE_H
#include <stdlib.h>
#include <string.h>

//#ifndef _WIN32
//#include <stdint.h>
//typedef uint64_t OCTCODE;
//#else
typedef __int64  OCTCODE;
//typedef int  OCTCODE;
//#endif

#define MAX_LEVEL 20

class OctCode
{
public:
	OctCode(float min[3], float max[3]);
	void setBoundingBox(float min[3], float max[3]);
	OCTCODE zOrder(float *p);
	OCTCODE hilbert(float *p);
private:
	float bbMin[3];
	float bbMax[3];
};

inline OctCode::OctCode(float min[3], float max[3])
{
	this->setBoundingBox(min, max);
}

inline void OctCode::setBoundingBox(float min[3], float max[3])
{
	memcpy(this->bbMin, min, sizeof(this->bbMin));
	memcpy(this->bbMax, max, sizeof(this->bbMax));
}

inline OCTCODE OctCode::zOrder(float *p)
{
	/*
	000   000
	100   001
	010   010
	110   011
	001   100
	101   101
	011   110
	111   111
	*/
	float min[3], max[3];
	memcpy(min, this->bbMin, sizeof(min));
	memcpy(max, this->bbMax, sizeof(max));
	register OCTCODE allNode = 0;
	for (int i = MAX_LEVEL; i >= 0; --i) {
		allNode <<= 3;
		if (p[0] + p[0]>min[0] + max[0]) {
			allNode |= 1;
			min[0] = (min[0] + max[0])*0.5;
		}
		else
			max[0] = (min[0] + max[0])*0.5;
		if (p[1] + p[1]>min[1] + max[1]) {
			allNode |= 2;
			min[1] = (min[1] + max[1])*0.5;
		}
		else
			max[1] = (min[1] + max[1])*0.5;
		if (p[2] + p[2]>min[2] + max[2]) {
			allNode |= 4;
			min[2] = (min[2] + max[2])*0.5;
		}
		else
			max[2] = (min[2] + max[2])*0.5;
	}
	return allNode;
}

inline OCTCODE OctCode::hilbert(float *p)
{
	/*
	000   000
	100   001
	010   010
	110   011
	001   100
	101   101
	011   110
	111   111
	*/
	float min[3], max[3];
	memcpy(min, this->bbMin, sizeof(min));
	memcpy(max, this->bbMax, sizeof(max));
	register OCTCODE allNode = 0;
	int curConvert = 0;
	int convert[2][8] = { { 1, 2, 6, 5, 0, 3, 7, 4 },{ 1, 2, 6, 5, 0, 3, 7, 4 } };
	// Map from Z-order to Hilbert curve
	static const int reflect[3][8] = {
		{ 1, 0, 3, 2, 5, 4, 7, 6 }, //X
		{ 4, 5, 6, 7, 0, 1, 2, 3 }, //Y
		{ 2, 3, 0, 1, 6, 7, 4, 5 }  //Z
	};
	static const int rotate[3][8] = {
		{ 1, 6, 5, 2, 3, 4, 7, 0 }, //X
		{ 7, 6, 1, 0, 3, 2, 5, 4 }, //Y
		{ 3, 0, 1, 2, 5, 6, 7, 4 }  //Z
	};
	for (int i = MAX_LEVEL; i >= 0; --i) {
		OCTCODE curNode = 0;
		allNode <<= 3;
		if (p[0] + p[0]>min[0] + max[0]) {
			curNode |= 1;
			min[0] = (min[0] + max[0])*0.5;
		}
		else
			max[0] = (min[0] + max[0])*0.5;
		if (p[1] + p[1]>min[1] + max[1]) {
			curNode |= 2;
			min[1] = (min[1] + max[1])*0.5;
		}
		else
			max[1] = (min[1] + max[1])*0.5;
		if (p[2] + p[2]>min[2] + max[2]) {
			curNode |= 4;
			min[2] = (min[2] + max[2])*0.5;
		}
		else
			max[2] = (min[2] + max[2])*0.5;
		curNode = convert[curConvert][curNode];
		allNode += curNode;
		switch (curNode) {
		case 0:
			// Rotate Y, Rotate Z
			for (int j = 0; j<8; j++)
				convert[1 - curConvert][j] = rotate[2][rotate[1][convert[curConvert][j]]];
			break;
		case 1:
		case 2:
			// Rotate X, Rotate Z, Reflect X, Reflect Y
			for (int j = 0; j<8; j++)
				convert[1 - curConvert][j] = reflect[1][reflect[0][rotate[2][rotate[0][convert[curConvert][j]]]]];
			break;
		case 3:
		case 4:
			// Rotate Z, Rotate Z
			for (int j = 0; j<8; j++)
				convert[1 - curConvert][j] = rotate[2][rotate[2][convert[curConvert][j]]];
			break;
		case 5:
		case 6:
			// Rotate Y, Rotate X
			for (int j = 0; j<8; j++)
				convert[1 - curConvert][j] = rotate[0][rotate[1][convert[curConvert][j]]];
			break;
		case 7:
			// Rotate Y, Rotate Z, Rotate X, Rotate X
			for (int j = 0; j<8; j++)
				convert[1 - curConvert][j] = rotate[0][rotate[0][rotate[2][rotate[1][convert[curConvert][j]]]]];
			break;
		default:
			abort();
		}
		curConvert = 1 - curConvert;
	}
	return allNode;
}

#endif
