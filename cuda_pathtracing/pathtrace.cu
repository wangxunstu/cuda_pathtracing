#include"pathtrace.h"

#include"cutil_math.h"
#include"glm/glm.hpp"
#include"glm/gtx/norm.hpp"


#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__


#include"geometry.h"

#include<cuda_texture_types.h>

#include <curand.h>
#include <curand_kernel.h>

#include <texture_fetch_functions.h>
#include<cuda_texture_types.h>
#include <device_launch_parameters.h>

#define M_PI 3.14159265359f  // pi
#define samps 512 // samples 

#define bounce 4

#define NUDGE_FACTOR     1e-3f  // epsilon
#define BVH_STACK_SIZE 32

// __device__ : executed on the device (GPU) and callable only from the device

static float3* dev_image = NULL;
static float3* host_image = NULL;

// Textures for vertices, triangles and BVH data
// (see CudaRender() below, as well as main() to see the data setup process)
texture<uint1, 1, cudaReadModeElementType> g_triIdxListTexture;
texture<float2, 1, cudaReadModeElementType> g_pCFBVHlimitsTexture;
texture<uint4, 1, cudaReadModeElementType>  g_pCFBVHindexesOrTrilistsTexture;
texture<float4, 1, cudaReadModeElementType> g_trianglesTexture;



struct Ray {
    float3 orig; 
    float3 dir;   
    __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR ,METAL};

struct Sphere 
{

    float rad;            // radius 
    float3 pos, emi, col; // position, emission, colour 
    Refl_t refl;          // reflection type (e.g. diffuse)

    __device__ float intersect_sphere(const Ray& r) const 
    {
        float3 op = pos - r.orig;    // distance from ray.orig to center sphere 
        float t, epsilon = 0.0001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.dir);    // b in quadratic equation
        float disc = b * b - dot(op, op) + rad * rad;  // discriminant quadratic equation
        if (disc < 0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
        else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // pick closest point in front of ray origin
    }
};



__device__ Sphere spheres[] = {

    { 16.f, { 0.0f,100.8, 0 }, { 6, 4, 2 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30  X: links rechts Y: op neer

    { 10000, { 50.0f, 40.8f, -1060 }, { 0.51, 0.51, 0.51 }, { 0.175f, 0.175f, 0.25f }, DIFF },

    { 100000, { 0.0f, -100001.2, 0 }, { 0, 0, 0 }, { 0.3f, 0.3f, 0.3f }, DIFF }, // double shell to prevent light leaking
};


__device__ inline bool intersect_scene(const Ray& r, float& t, int& id) 
{

    float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;)  
        if ((d = spheres[i].intersect_sphere(r)) && d < t) { 
            t = d;  
            id = i; 
        }
    return t < inf;
}

__device__ static float getrandom(unsigned int* seed0, unsigned int* seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}


// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const Vector3Df& originInWorldSpace, const Vector3Df& rayInWorldSpace, int boxIdx)
{
    // set Tnear = - infinity, Tfar = infinity
    //
    // For each pair of planes P associated with X, Y, and Z do:
    //     (example using X planes)
    //     if direction Xd = 0 then the ray is parallel to the X planes, so
    //         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
    //             return false
    //     else, if the ray is not parallel to the plane then
    //     begin
    //         compute the intersection distance of the planes
    //         T1 = (Xl - Xo) / Xd
    //         T2 = (Xh - Xo) / Xd
    //         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
    //         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
    //         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
    //         If Tnear > Tfar box is missed so
    //             return false
    //         If Tfar < 0 box is behind ray
    //             return false
    //     end
    // end of for loop

    float Tnear, Tfar;
    Tnear = -FLT_MAX;
    Tfar = FLT_MAX;

    float2 limits;

    // box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {						    \
	if (originInWorldSpace.##c < limits.x) return false;					    \
	if (originInWorldSpace.##c > limits.y) return false;					    \
	} else {											    \
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return false;									    \
	if (Tfar < 0.f)	return false;									    \
	}

    limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx); // box.bottom._x/top._x placed in limits.x/limits.y
    //limits = make_float2(cudaBVHlimits[6 * boxIdx + 0], cudaBVHlimits[6 * boxIdx + 1]);
    CHECK_NEAR_AND_FAR_INTERSECTION(x)
        limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 1); // box.bottom._y/top._y placed in limits.x/limits.y
        //limits = make_float2(cudaBVHlimits[6 * boxIdx + 2], cudaBVHlimits[6 * boxIdx + 3]);
    CHECK_NEAR_AND_FAR_INTERSECTION(y)
        limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 2); // box.bottom._z/top._z placed in limits.x/limits.y
        //limits = make_float2(cudaBVHlimits[6 * boxIdx + 4], cudaBVHlimits[6 * boxIdx + 5]);
    CHECK_NEAR_AND_FAR_INTERSECTION(z)

        // If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
        return true;
}

//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

__device__ bool BVH_IntersectTriangles(
    int* cudaBVHindexesOrTrilists, const Vector3Df& origin, const Vector3Df& ray, unsigned avoidSelf,
    int& pBestTriIdx, Vector3Df& pointHitInWorldSpace, float& kAB, float& kBC, float& kCA, float& hitdist,
    float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, Vector3Df& boxnormal)
{
    // in the loop below, maintain the closest triangle and the point where we hit it:
    pBestTriIdx = -1;
    float bestTriDist;

    // start from infinity
    bestTriDist = FLT_MAX;

    // create a stack for each ray
    // the stack is just a fixed size array of indices to BVH nodes
    int stack[BVH_STACK_SIZE];

    int stackIdx = 0;
    stack[stackIdx++] = 0;
    Vector3Df hitpoint;

    // while the stack is not empty
    while (stackIdx) {

        // pop a BVH node (or AABB, Axis Aligned Bounding Box) from the stack
        int boxIdx = stack[stackIdx - 1];
        //uint* pCurrent = &cudaBVHindexesOrTrilists[boxIdx]; 

        // decrement the stackindex
        stackIdx--;

        // fetch the data (indices to childnodes or index in triangle list + trianglecount) associated with this node
        uint4 data = tex1Dfetch(g_pCFBVHindexesOrTrilistsTexture, boxIdx);

        // original, "pure" BVH form...
        //if (!pCurrent->IsLeaf()) {

        // cache-friendly BVH form...
        //if (!(cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x80000000)) { // INNER NODE

        // texture memory BVH form...

        // determine if BVH node is an inner node or a leaf node by checking the highest bit (bitwise AND operation)
        // inner node if highest bit is 1, leaf node if 0

        if (!(data.x & 0x80000000)) {   // INNER NODE

            // if ray intersects inner node, push indices of left and right child nodes on the stack
            if (RayIntersectsBox(origin, ray, boxIdx)) {

                //stack[stackIdx++] = pCurrent->u.inner._idxRight;
                //stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 1];
                stack[stackIdx++] = data.y; // right child node index

                //stack[stackIdx++] = pCurrent->u.inner._idxLeft;
                //stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 2];
                stack[stackIdx++] = data.z; // left child node index

                // return if stack size is exceeded
                if (stackIdx > BVH_STACK_SIZE)
                {
                    return false;
                }
            }
        }
        else { // LEAF NODE

            // original, "pure" BVH form...
            // BVHLeaf *p = dynamic_cast<BVHLeaf*>(pCurrent);
            // for(std::list<const Triangle*>::iterator it=p->_triangles.begin();
            //    it != p->_triangles.end();
            //    it++)

            // cache-friendly BVH form...
            // for(unsigned i=pCurrent->u.leaf._startIndexInTriIndexList;
            //    i<pCurrent->u.leaf._startIndexInTriIndexList + (pCurrent->u.leaf._count & 0x7fffffff);

            // texture memory BVH form...
            // for (unsigned i = cudaBVHindexesOrTrilists[4 * boxIdx + 3]; i< cudaBVHindexesOrTrilists[4 * boxIdx + 3] + (cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x7fffffff); i++) { // data.w = number of triangles in leaf

            // loop over every triangle in the leaf node
            // data.w is start index in triangle list
            // data.x stores number of triangles in leafnode (the bitwise AND operation extracts the triangle number)
            for (unsigned i = data.w; i < data.w + (data.x & 0x7fffffff); i++) 
            {

                // original, "pure" BVH form...
                //const Triangle& triangle = *(*it);

                // cache-friendly BVH form...
                //const Triangle& triangle = pTriangles[cudaTriIdxList[i]];

                // texture memory BVH form...
                // fetch the index of the current triangle
                int idx = tex1Dfetch(g_triIdxListTexture, i).x;
                //int idx = cudaTriIdxList[i];

                // check if triangle is the same as the one intersected by previous ray
                // to avoid self-reflections/refractions
                if (avoidSelf == idx)
                    continue;

                // fetch triangle center and normal from texture memory
                float4 center = tex1Dfetch(g_trianglesTexture, 5 * idx);
                float4 normal = tex1Dfetch(g_trianglesTexture, 5 * idx + 1);

                // use the pre-computed triangle intersection data: normal, d, e1/d1, e2/d2, e3/d3
                float k = dot(normal, ray);
                if (k == 0.0f)
                    continue; // this triangle is parallel to the ray, ignore it.

                float s = (normal.w - dot(normal, origin)) / k;
                if (s <= 0.0f) // this triangle is "behind" the origin.
                    continue;
                if (s <= NUDGE_FACTOR)  // epsilon
                    continue;
                Vector3Df hit = ray * s;
                hit += origin;

                // ray triangle intersection
                // Is the intersection of the ray with the triangle's plane INSIDE the triangle?

                float4 ee1 = tex1Dfetch(g_trianglesTexture, 5 * idx + 2);
                //float4 ee1 = make_float4(cudaTriangleIntersectionData[20 * idx + 8], cudaTriangleIntersectionData[20 * idx + 9], cudaTriangleIntersectionData[20 * idx + 10], cudaTriangleIntersectionData[20 * idx + 11]);
                float kt1 = dot(ee1, hit) - ee1.w;
                if (kt1 < 0.0f) continue;

                float4 ee2 = tex1Dfetch(g_trianglesTexture, 5 * idx + 3);
                //float4 ee2 = make_float4(cudaTriangleIntersectionData[20 * idx + 12], cudaTriangleIntersectionData[20 * idx + 13], cudaTriangleIntersectionData[20 * idx + 14], cudaTriangleIntersectionData[20 * idx + 15]);
                float kt2 = dot(ee2, hit) - ee2.w;
                if (kt2 < 0.0f) continue;

                float4 ee3 = tex1Dfetch(g_trianglesTexture, 5 * idx + 4);
                //float4 ee3 = make_float4(cudaTriangleIntersectionData[20 * idx + 16], cudaTriangleIntersectionData[20 * idx + 17], cudaTriangleIntersectionData[20 * idx + 18], cudaTriangleIntersectionData[20 * idx + 19]);
                float kt3 = dot(ee3, hit) - ee3.w;
                if (kt3 < 0.0f) continue;

                // ray intersects triangle, "hit" is the world space coordinate of the intersection.
                {
                    // is this intersection closer than all the others?
                    float hitZ = distancesq(origin, hit);
                    if (hitZ < bestTriDist) {

                        // maintain the closest hit
                        bestTriDist = hitZ;
                        hitdist = sqrtf(bestTriDist);
                        pBestTriIdx = idx;
                        pointHitInWorldSpace = hit;

                        // store barycentric coordinates (for texturing, not used for now)
                        kAB = kt1;
                        kBC = kt2;
                        kCA = kt3;
                    }
                }
            }
        }
    }

    return pBestTriIdx != -1;
}

__device__ float3 radiance(Ray& r, int avoidSelf, unsigned int* s1, unsigned int* s2,
    Triangle* cudaTriangles, int* cudaBVHindexedOrTriLists,
    float* cudaBVHLimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList)
{ 

    r.dir = normalize(r.dir);

    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    int sphere_id   = -1;
    int triangle_id = -1;
    int pBestTriIdx = -1;
    int geomtype    = 1;


    Refl_t refltype  = DIFF;

    for (int bounces = 0; bounces < bounce; bounces++) 
    { 

       float t;           // distance to closest intersection 
       int id = 0;        // index of closest intersected sphere 

       float3 col;

       float hit_distance_bvh = 1e20;
       float hit_distance_sphere = 1e20;

       float scene_t = 1e20;

       const Triangle* pBestTri = NULL;
       Vector3Df pointHitInWorldSpace;
       float kAB = 0.f, kBC = 0.f, kCA = 0.f; // distances from the 3 edges of the triangle (from where we hit it), to be used for texturing

       float tmin = 1e20;
       float tmax = -1e20;
       float inf = 1e20;
       Vector3Df f = Vector3Df(0, 0, 0);
       Vector3Df emit = Vector3Df(0, 0, 0);
       float3 x; // intersection point
       float3 n; // normal
       float3 nl; // oriented normal
       Vector3Df boxnormal = Vector3Df(0, 0, 0);
       Vector3Df dw; // ray direction of next path segment


            
       Vector3Df _ray_ori = Vector3Df(r.orig.x, r.orig.y, r.orig.z);
       Vector3Df _ray_dir = Vector3Df(r.dir.x,  r.dir.y,  r.dir.z);

       BVH_IntersectTriangles(
               cudaBVHindexedOrTriLists, _ray_ori, _ray_dir, avoidSelf,
               pBestTriIdx, pointHitInWorldSpace, kAB, kBC, kCA, hit_distance_bvh, cudaBVHLimits,
               cudaTriangleIntersectionData, cudaTriIdxList, boxnormal);


       avoidSelf = pBestTriIdx;

        float numspheres = sizeof(spheres) / sizeof(Sphere);
        for (int i = int(numspheres); i--;) 
        { 
            if ((hit_distance_sphere = spheres[i].intersect_sphere(Ray(r.orig, r.dir))) && hit_distance_sphere < scene_t)
            {
                scene_t = hit_distance_sphere; 
                sphere_id = i; 
                geomtype = 1;
            }
        }


        if (hit_distance_bvh < scene_t && hit_distance_bvh > 0.002)
        {
            scene_t = hit_distance_bvh;
            triangle_id = pBestTriIdx;
            geomtype = 2;
        }
             
            
        t = scene_t;

        if (geomtype == 1 )
        {
            Sphere& sphere = spheres[sphere_id]; // hit object with closest intersection

            const Sphere& obj = spheres[sphere_id];  // hitobject
            x = r.orig + r.dir * t;          // hitpoint 
            n = normalize(x - obj.pos);    // normal
            nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal

            accucolor += mask * obj.emi;

            col = obj.col;

            refltype = DIFF;
        }


        // TRIANGLES:5
        if (geomtype == 2)
        {

            pBestTri = &cudaTriangles[triangle_id];

            x = make_float3(pointHitInWorldSpace.x, pointHitInWorldSpace.y, pointHitInWorldSpace.z);  // intersection point
            n = make_float3(pBestTri->_normal.x, pBestTri->_normal.y, pBestTri->_normal.z);  // normal
           
            n = normalize(n);
            nl = dot(n, make_float3(r.dir.x, r.dir.y, r.dir.z)) < 0 ? n : n * -1.f;  // correctly oriented normal

            col = make_float3(0.9f, 0.3f, 0.0f);


            refltype = METAL;
        }

        if (refltype == METAL)
        {
            // compute random perturbation of ideal reflection vector
            // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
            float phi = 2 * M_PI * getrandom(s1,s2);
            float r2 = getrandom(s2, s1);
            float phongexponent = 100;
            float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
            float sinTheta = sqrtf(1 - cosTheta * cosTheta);

            // create orthonormal basis uvw around reflection vector with hitpoint as origin 
            // w is ray direction for ideal reflection

            Vector3Df rayInWorldSpace = Vector3Df(r.dir.x, r.dir.y, r.dir.z);
            Vector3Df normal = Vector3Df(n.x, n.y, n.z);
            Vector3Df w = rayInWorldSpace - normal * 2.0f * dot(normal, rayInWorldSpace); w.normalize();
            Vector3Df u = cross((fabs(w.x) > .1 ? Vector3Df(0, 1, 0) : Vector3Df(1, 0, 0)), w); u.normalize();
            Vector3Df v = cross(w, u); // v is normalised by default

            // compute cosine weighted random ray direction on hemisphere 
            dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
            dw.normalize();

            // offset origin next path segment to prevent self intersection
            r.orig = x;//+ w * 0.01;  // scene size dependent
            r.dir = make_float3(dw.x,dw.y,dw.z);

            // multiply mask with colour of object
            mask *= col;
        }


        if (refltype == DIFF)
        {
            float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
            float r2 = getrandom(s1, s2);  // pick random number for elevation
            float r2s = sqrtf(r2);

            // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction 
            // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // compute random ray direction on hemisphere using polar coordinates
            // cosine weighted importance sampling (favours ray directions closer to normal direction)
            float3 new_d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

            // new ray origin is intersection point of previous ray with scene
            r.orig = x + nl * 0.05f; // offset ray origin slightly to prevent self intersection
            r.dir = new_d;

            mask *= col;    // multiply with colour of object       
            mask *= dot(new_d, nl);  // weigh light contribution using cosine of angle between incident light and normal
            mask *= 2;          // fudge factor

        }

    }

    return accucolor;
}



// __global__ : executed on the device (GPU) and callable only from host (CPU) 
// this kernel runs in parallel on all the CUDA threads

__global__ void render_kernel(float3* output, int width, int height, Triangle* cudaTriangles, int* cudaBVHindexedOrTriLists,
    float* cudaBVHLimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList)
{

    // assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1) * width + x; // index of current pixel (calculated using thread index) 

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

    // generate ray directed at lower left corner of the screen
    // compute directions for all other rays by adding cx and cy increments in x and y direction
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction) 
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color       

    r = make_float3(0.0f); // reset r to zero for every pixel 

    float inv_samples = 1.f / float(samps);

    for (int sx = 0; sx < 2; sx++)
    {
        for (int sy = 0; sy < 2; sy++)
        {
            float _r1 = 2.f*getrandom(&s1, &s2);
            float _r2 = 2.f*getrandom(&s2, &s1);

            float dx = _r1 < 1.f ? sqrtf(_r1) - 1.f : 1.f - sqrtf(2.f - _r1);
            float dy = _r2 < 1.f ? sqrtf(_r2) - 1.f : 1.f - sqrtf(2.f - _r2);


            for (int s = 0; s < samps; s++) {  // samples per pixel

               //compute primary ray direction
                float3 d = cam.dir + 
                           cx * (((sx + 0.5f + dx)/2.f +x) / width - .5) +
                           cy * (((sy + 0.5f + dx)/2.f +y) / height - .5);

                // create primary ray, add incoming radiance to pixelcolor
                Ray _ray(cam.orig + d * 40, normalize(d));
                r = r + radiance(_ray, -1, &s1, &s2, cudaTriangles, cudaBVHindexedOrTriLists, cudaBVHLimits,
                    cudaTriangleIntersectionData, cudaTriIdxList) * inv_samples;
            }       // Camera rays are pushed ^^^^^ forward to start in interior 
        }
    }

    r *= 0.25;

    // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
    output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}


__host__ __device__ float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

__host__ __device__ int toInt(float x) { return int(powf(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction



//Kernel that writes  image to the OpenGL PBO directly.

__global__ void sendImageToPBO(uchar4* pbo, int _width ,int _height, float3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;


    if (x < _width && y < _height) 
    {
        int index = x + (y * _width);

        float3 pix = image[index];

        glm::ivec3 color;
        color.x = toInt(pix.x);
        color.y = toInt(pix.y);
        color.z = toInt(pix.z);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

void pathtrace(uchar4* pbo, Triangle* cudaTriangles, int* cudaBVHindexedOrTriLists,
    float* cudaBVHLimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList)
{

    host_image = new float3[width * height]; // pointer to memory for image on the host (system RAM)

    // allocate memory on the CUDA device (GPU VRAM)
    cudaMalloc(&dev_image, width * height * sizeof(float3));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    {

        cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<uint1>();
        cudaBindTexture(NULL, &g_triIdxListTexture, cudaTriIdxList, &channel1desc, g_triIndexListNo * sizeof(uint1));

        cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float2>();
        cudaBindTexture(NULL, &g_pCFBVHlimitsTexture, cudaBVHLimits, &channel2desc, g_pCFBVH_No * 6 * sizeof(float));

        cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<uint4>();
        cudaBindTexture(NULL, &g_pCFBVHindexesOrTrilistsTexture, cudaBVHindexedOrTriLists, &channel3desc,
            g_pCFBVH_No * sizeof(uint4));

        cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
        cudaBindTexture(NULL, &g_trianglesTexture, cudaTriangleIntersectionData, &channel5desc, 
            g_trianglesNo * 20 * sizeof(float));
    }


    // schedule threads on device and launch CUDA kernel from host
    render_kernel << < grid, block >> > (dev_image, width, height, cudaTriangles, cudaBVHindexedOrTriLists,
        cudaBVHLimits,cudaTriangleIntersectionData,cudaTriIdxList);

    // copy results of computation from device back to host
   // cudaMemcpy(host_image, dev_image, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    sendImageToPBO<< <grid, block >> >(pbo, width, height, dev_image);

    printf("send image to pbo finish");

    // free CUDA memory
    cudaFree(dev_image);

}