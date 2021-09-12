#pragma once
#include <vector>
#include <vector_types.h>
#include "geometry.h"
#include "bvh.h"

extern int width;
extern int height;

// global variables
extern unsigned g_verticesNo;
extern Vertex* g_vertices;
extern unsigned g_trianglesNo;
extern Triangle* g_triangles;
extern BVHNode* g_pSceneBVH;
extern unsigned g_triIndexListNo;
extern int* g_triIndexList;
extern unsigned g_pCFBVH_No;
extern CacheFriendlyBVHNode* g_pCFBVH;

void pathtraceFree();
void pathtrace(uchar4* pbo, Triangle* cudaTriangles, int* cudaBVHindexedOrTriLists,
	float* cudaBVHLimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList);
