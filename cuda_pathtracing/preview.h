#pragma once

#pragma warning(disable:C4996)

#include "geometry.h"
#include "bvh.h"


extern GLuint pbo;

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


std::string currentTimeString();

bool init();
void mainLoop();
