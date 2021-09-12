
#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#pragma warning(disable:4996)
#define IMGUI_IMPL_OPENGL_LOADER_GLEW


#include "loader.h"
#include "bvh.h"

#include "geometry.h"

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;

GLuint displayImage;

int width = 760, height = 540;

// CUDA arrays
Vertex*   cudaVertices2 = NULL;
Triangle* cudaTriangles2 = NULL;
float*    cudaTriangleIntersectionData2 = NULL;
int*      cudaTriIdxList2 = NULL;
float*    cudaBVHlimits2 = NULL;
int*      cudaBVHindexesOrTrilists2 = NULL;



GLFWwindow* window;

void initTextures() 
{
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void initVAO(void) 
{
	GLfloat vertices[] = 
	{
		 1.f,  1.f,
		 1.f, -1.f,
		-1.f, -1.f,
		-1.f,  1.f
	};

	GLfloat texcoords[] = 
	{
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
	};

	GLushort indices[] = { 0, 1, 3, 1, 2, 3};

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() 
{
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}



void errorCallback(int error, const char* description) 
{
	fprintf(stderr, "%s\n", description);
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) 
{

}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{

}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{

}

void preCudaScene()
{
	const char* file_name = "models/dragon.ply";
	float maxi = load_object(file_name);

	UpdateBoundingVolumeHierarchy(file_name);

	// store vertices in a GPU friendly format using float4
	float* pVerticesData = (float*)malloc(g_verticesNo * 8 * sizeof(float));
	for (unsigned f = 0; f < g_verticesNo; f++) {

		// first float4 stores vertex xyz position and precomputed ambient occlusion
		pVerticesData[f * 8 + 0] = g_vertices[f].x;
		pVerticesData[f * 8 + 1] = g_vertices[f].y;
		pVerticesData[f * 8 + 2] = g_vertices[f].z;
		pVerticesData[f * 8 + 3] = g_vertices[f]._ambientOcclusionCoeff;
		// second float4 stores vertex normal xyz
		pVerticesData[f * 8 + 4] = g_vertices[f]._normal.x;
		pVerticesData[f * 8 + 5] = g_vertices[f]._normal.y;
		pVerticesData[f * 8 + 6] = g_vertices[f]._normal.z;
		pVerticesData[f * 8 + 7] = 0.f;
	}

	// copy vertex data to CUDA global memory
	cudaMalloc((void**)&cudaVertices2, g_verticesNo * 8 * sizeof(float));
	cudaMemcpy(cudaVertices2, pVerticesData, g_verticesNo * 8 * sizeof(float), cudaMemcpyHostToDevice);


	// store precomputed triangle intersection data in a GPU friendly format using float4
	float* pTrianglesIntersectionData = (float*)malloc(g_trianglesNo * 20 * sizeof(float));

	for (unsigned e = 0; e < g_trianglesNo; e++) {
		// Texture-wise:
		//
		// first float4, triangle center + two-sided bool
		pTrianglesIntersectionData[20 * e + 0] = g_triangles[e]._center.x;
		pTrianglesIntersectionData[20 * e + 1] = g_triangles[e]._center.y;
		pTrianglesIntersectionData[20 * e + 2] = g_triangles[e]._center.z;
		pTrianglesIntersectionData[20 * e + 3] = g_triangles[e]._twoSided ? 1.0f : 0.0f;
		// second float4, normal
		pTrianglesIntersectionData[20 * e + 4] = g_triangles[e]._normal.x;
		pTrianglesIntersectionData[20 * e + 5] = g_triangles[e]._normal.y;
		pTrianglesIntersectionData[20 * e + 6] = g_triangles[e]._normal.z;
		pTrianglesIntersectionData[20 * e + 7] = g_triangles[e]._d;
		// third float4, precomputed plane normal of triangle edge 1
		pTrianglesIntersectionData[20 * e + 8] = g_triangles[e]._e1.x;
		pTrianglesIntersectionData[20 * e + 9] = g_triangles[e]._e1.y;
		pTrianglesIntersectionData[20 * e + 10] = g_triangles[e]._e1.z;
		pTrianglesIntersectionData[20 * e + 11] = g_triangles[e]._d1;
		// fourth float4, precomputed plane normal of triangle edge 2
		pTrianglesIntersectionData[20 * e + 12] = g_triangles[e]._e2.x;
		pTrianglesIntersectionData[20 * e + 13] = g_triangles[e]._e2.y;
		pTrianglesIntersectionData[20 * e + 14] = g_triangles[e]._e2.z;
		pTrianglesIntersectionData[20 * e + 15] = g_triangles[e]._d2;
		// fifth float4, precomputed plane normal of triangle edge 3
		pTrianglesIntersectionData[20 * e + 16] = g_triangles[e]._e3.x;
		pTrianglesIntersectionData[20 * e + 17] = g_triangles[e]._e3.y;
		pTrianglesIntersectionData[20 * e + 18] = g_triangles[e]._e3.z;
		pTrianglesIntersectionData[20 * e + 19] = g_triangles[e]._d3;
	}


	// copy precomputed triangle intersection data to CUDA global memory
	cudaMalloc((void**)&cudaTriangleIntersectionData2, g_trianglesNo * 20 * sizeof(float));
	cudaMemcpy(cudaTriangleIntersectionData2, pTrianglesIntersectionData, g_trianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice);

	// copy triangle data to CUDA global memory
	cudaMalloc((void**)&cudaTriangles2, g_trianglesNo * sizeof(Triangle));
	cudaMemcpy(cudaTriangles2, g_triangles, g_trianglesNo * sizeof(Triangle), cudaMemcpyHostToDevice);


	// Allocate CUDA-side data (global memory for corresponding textures) for Bounding Volume Hierarchy data
// See BVH.h for the data we are storing (from CacheFriendlyBVHNode)

// Leaf nodes triangle lists (indices to global triangle list)
// copy triangle indices to CUDA global memory
	cudaMalloc((void**)&cudaTriIdxList2, g_triIndexListNo * sizeof(int));
	cudaMemcpy(cudaTriIdxList2, g_triIndexList, g_triIndexListNo * sizeof(int), cudaMemcpyHostToDevice);

	// Bounding box limits need bottom._x, top._x, bottom._y, top._y, bottom._z, top._z...
	// store BVH bounding box limits in a GPU friendly format using float2
	float* pLimits = (float*)malloc(g_pCFBVH_No * 6 * sizeof(float));

	for (unsigned h = 0; h < g_pCFBVH_No; h++) {
		// Texture-wise:
		// First float2
		pLimits[6 * h + 0] = g_pCFBVH[h]._bottom.x;
		pLimits[6 * h + 1] = g_pCFBVH[h]._top.x;
		// Second float2
		pLimits[6 * h + 2] = g_pCFBVH[h]._bottom.y;
		pLimits[6 * h + 3] = g_pCFBVH[h]._top.y;
		// Third float2
		pLimits[6 * h + 4] = g_pCFBVH[h]._bottom.z;
		pLimits[6 * h + 5] = g_pCFBVH[h]._top.z;
	}

	// copy BVH limits to CUDA global memory
	cudaMalloc((void**)&cudaBVHlimits2, g_pCFBVH_No * 6 * sizeof(float));
	cudaMemcpy(cudaBVHlimits2, pLimits, g_pCFBVH_No * 6 * sizeof(float), cudaMemcpyHostToDevice);

	// ..and finally, from CacheFriendlyBVHNode, the 4 integer values:
	// store BVH node attributes (triangle count, startindex, left and right child indices) in a GPU friendly format using uint4
	int* pIndexesOrTrilists = (int*)malloc(g_pCFBVH_No * 4 * sizeof(unsigned));

	for (unsigned g = 0; g < g_pCFBVH_No; g++) {
		// Texture-wise:
		// A single uint4
		pIndexesOrTrilists[4 * g + 0] = g_pCFBVH[g].u.leaf._count;  // number of triangles stored in this node if leaf node
		pIndexesOrTrilists[4 * g + 1] = g_pCFBVH[g].u.inner._idxRight; // index to right child if inner node
		pIndexesOrTrilists[4 * g + 2] = g_pCFBVH[g].u.inner._idxLeft;  // index to left node if inner node
		pIndexesOrTrilists[4 * g + 3] = g_pCFBVH[g].u.leaf._startIndexInTriIndexList; // start index in list of triangle indices if leaf node
		// union

	}

	// copy BVH node attributes to CUDA global memory
	cudaMalloc((void**)&cudaBVHindexesOrTrilists2, g_pCFBVH_No * 4 * sizeof(unsigned));
	cudaMemcpy(cudaBVHindexesOrTrilists2, pIndexesOrTrilists, g_pCFBVH_No * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);

	// Initialisation Done!
	std::cout << "Rendering data initialised and copied to CUDA global memory\n";

}

bool init()
{
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) 
	{
		exit(EXIT_FAILURE);
	}
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	window = glfwCreateWindow(width, height, "CUDA Path Tracer", NULL, NULL);
	if (!window) 
	{
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSwapInterval(1); // Enable vsync

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	preCudaScene();

	// Initialize other stuff
	initVAO();
	initTextures();
	initPBO();

	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	{

		// Setup Dear ImGui context
		//IMGUI_CHECKVERSION();
		//ImGui::CreateContext();

		// Setup Dear ImGui style
		//ImGui::StyleColorsDark();

		// Setup Platform/Renderer bindings
		//ImGui_ImplGlfw_InitForOpenGL(window, true);
		//ImGui_ImplOpenGL3_Init(glsl_version);
	}

	return true;
}


void runCuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	uchar4* pbo_dptr = NULL;
	cudaGLMapBufferObject((void**)&pbo_dptr, pbo);
	pathtrace(pbo_dptr,cudaTriangles2,cudaBVHindexesOrTrilists2,
		cudaBVHlimits2,cudaTriangleIntersectionData2
	,cudaTriIdxList2);        // execute the kernel
	cudaGLUnmapBufferObject(pbo);
}


void mainLoop() 
{

	runCuda();

	glClearColor(0.2f, 0.2f, 0.3f, 1.0f);

	while (!glfwWindowShouldClose(window)) 
	{
		glfwPollEvents();


		std::string title = "CUDA Path Tracer";
		glfwSetWindowTitle(window, title.c_str());
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

		// Draw imgui
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
	//	drawGui(display_w, display_h);

		// Display content
		glViewport(0,0,width,height);
		glfwSwapBuffers(window);
	}

	// Cleanup
	//ImGui_ImplOpenGL3_Shutdown();
	//ImGui_ImplGlfw_Shutdown();
	//ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
