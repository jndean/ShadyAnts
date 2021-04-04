#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>


#define PI 3.141592653589793


// Agents
#define NUM_AGENTS 1000
#define SPEED 1.5f

// Trails
#define DIFFUSE_RAD 1
#define DECAY 0.7f
#define FLUX 0.2f


struct Agent {
    float x, y;
    float angle;
    uint32_t state;
};


Agent* d_agents;
float* d_trails_back;
float* d_trails_front;


// ------------------------ HELPERS ------------------------ //

__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

inline __device__
uint32_t step_state(uint32_t state)
{
    state ^= 2747636419u;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    return state;
}



// ------------------------------ SETUP ------------------------------ //



extern "C"
__host__ void CUDAinit(unsigned int** texture, int imgw, int imgh, int ubyte_size)
{
    unsigned int texture_size = imgw * imgh * 4 * ubyte_size;
    checkCudaErrors(cudaMalloc((void **)texture, texture_size));
    
    Agent h_agents[NUM_AGENTS];
    srand(6);
    for (int i = 0; i < NUM_AGENTS; ++i) {
	h_agents[i].x = rand() % imgw;
	h_agents[i].y = rand() % imgh;
	h_agents[i].state = rand();
	h_agents[i].angle = ((float)h_agents[i].state) / RAND_MAX * PI * 2;
    }
    checkCudaErrors(cudaMalloc(&d_agents, NUM_AGENTS * sizeof(Agent)));
    checkCudaErrors(cudaMalloc(&d_trails_back, imgw * imgh * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_trails_front, imgw * imgh * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_agents, h_agents, NUM_AGENTS * sizeof(Agent), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}


// ------------------------------ KERNELS ------------------------------ //


__global__ void
moveAgents(Agent* agents, float *trails, float imgw, float imgh)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= NUM_AGENTS) return;

    Agent agent = agents[i];

    agent.x += sin(agent.angle) * SPEED;
    agent.y += cos(agent.angle) * SPEED;
    if (agent.x < 0 || agent.x >= imgw || agent.y < 0 || agent.y >= imgh) {
	agent.state = step_state(agent.state);
	agent.angle = agent.state / 1000.0f;
	agent.x = clamp(agent.x, 0.0f, imgw-1);
	agent.y = clamp(agent.y, 0.0f, imgh-1);
    }
    
    trails[((int) agent.y) * (int)imgw + ((int) agent.x)] = 255.0f;
    agents[i] = agent;
}




__global__ void
diffuse(float* trails_front, float* trails_back, unsigned int *g_odata, int imgw, int imgh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgw || y >= imgh) return;

    float val = trails_front[y * imgw + x];
    g_odata[y*imgw+x] = rgbToInt(val, val, val);
    
    float sum = 0;
    for (int dy = -DIFFUSE_RAD; dy < DIFFUSE_RAD; ++dy) {
	for (int dx = -DIFFUSE_RAD; dx < DIFFUSE_RAD; ++dx) {
	    int sample_y = clamp(y+dy, 0, imgh);
	    int sample_x = clamp(x+dx, 0, imgw);
	    sum += trails_front[sample_y * imgw + sample_x];
	}
    }
    sum /= 2; //DIFFUSE_AREA;
    
    trails_back[y*imgw+x] = FLUX * sum + (1-FLUX) * val * DECAY;
}



extern "C" void
step_simulation(unsigned int *odata, int imgw, int imgh)
{

    int block_size = 32;
    int num_blocks = (NUM_AGENTS + block_size - 1) / block_size;
    moveAgents<<<num_blocks, block_size>>>(d_agents, d_trails_front, imgw, imgh);

    
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((imgw + block_size - 1) / block_size, (imgh + block_size - 1) / block_size);
    diffuse<<<grid_dim, block_dim>>>(d_trails_front, d_trails_back, odata, imgw, imgh);

    std::swap(d_trails_front, d_trails_back);
    
}