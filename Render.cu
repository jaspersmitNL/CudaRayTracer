#include "Render.cuh"



__device__ Hitable** objects;
__device__ HitableList* world;

__device__ uint32_t ConvertToRGBA(const glm::vec4& color)
{
	uint8_t r = (uint8_t)(color.r * 255.0f);
	uint8_t g = (uint8_t)(color.g * 255.0f);
	uint8_t b = (uint8_t)(color.b * 255.0f);
	uint8_t a = (uint8_t)(color.a * 255.0f);

	uint32_t result = (a << 24) | (b << 16) | (g << 8) | r;
	return result;
}



__device__ glm::vec3 RandomInUnitSphere(curandState* randState)
{
		glm::vec3 p;
	do
	{
				p = 2.0f * glm::vec3(curand_uniform(randState), curand_uniform(randState), curand_uniform(randState)) - glm::vec3(1.0f);
	} while (glm::length(p) >= 1.0f);
	return p;
}

__device__ glm::vec3 RayColorNew(const Ray& ray, curandState* randState)
{
	if (world == nullptr)
	{
		return glm::vec3(0.0f);
	}

	Ray currentRay = ray;
	glm::vec3 currentAttenuation(1.0, 1.0, 1.0);
	for(int i =0; i < 50; i++)
	{
		HitRecord record;
		if(world->hit(currentRay, 0.001f,FLT_MAX, record))
		{
		Ray scattered;
			glm::vec3 attenuation;
			if(record.material->scatter(currentRay, record, attenuation, scattered, randState))
			{
								currentRay = scattered;
				currentAttenuation *= attenuation;
			} else
			{
								return glm::vec3(0.0f);
			}
		} else
		{
			glm::vec3 unit_direction = glm::normalize(currentRay.direction());
			float t = 0.5f * (unit_direction.y + 1.0f);
			glm::vec3 c = (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
			return currentAttenuation * c;
		}
	}
	return glm::vec3(0.0f);

}



__global__ void InitKernel(int width, int height)
{
	if (threadIdx.x != 0) return;
	printf("InitKernel(%i, %i)\n", width, height);


	objects = new Hitable*[4];
	objects[0] = new Sphere(glm::vec3(0, 0, -1), 0.5,
		new Lambertian(glm::vec3(0.8, 0.3, 0.3)));
	objects[1] = new Sphere(glm::vec3(0, -100.5, -1), 100,
		new Lambertian(glm::vec3(0.8, 0.8, 0.0)));
	objects[2] = new Sphere(glm::vec3(1, 0, -1), 0.5,
		new Metal(glm::vec3(0.8, 0.6, 0.2), 1.0));
	objects[3] = new Sphere(glm::vec3(-1, 0, -1), 0.5,
		new Metal(glm::vec3(0.8, 0.8, 0.8), 0.05));
	world = new HitableList(objects, 4);


}

__global__ void InitPerPixel(int width, int height, curandState* randState)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int index = x + y * width;

	curand_init(1984, index, 0, &randState[index]);

	
	  

}

__global__ void RenderKernel(int width, int height, uint32_t* pixels, Camera* camera, curandState* randState)
{
	int samples = 200;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width || j >= height) return;
	int index = i + j * width;
	curandState  rand = randState[index];
	glm::vec3 color = glm::vec3(0.0f);
	for(int s = 0; s < samples; s++)
	{
		float u = float(i + curand_uniform(&rand)) / float(width);
		float v = float(j + curand_uniform(&rand)) / float(height);
		u = u * 2.0f - 1.0f; // -1 -> 1
		v = v * 2.0f - 1.0f; // -1 -> 1

		Ray ray = camera->GetRay(glm::vec2(u, v));
		color += RayColorNew(ray, &rand);
	}
	pixels[index] = ConvertToRGBA(glm::vec4(color / float(samples), 1.0f));
}

void Render(glm::vec2 size, uint32_t* pixels, Camera* camera, curandState* randState)
{
	int width = size.x;
	int height = size.y;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	RenderKernel<<<numBlocks, threadsPerBlock>>>(width, height, pixels, camera, randState);
	cudaDeviceSynchronize();
}

void Init(int width, int height, curandState* randState)
{
	InitKernel<<<1, 1>>>(width, height);
	cudaDeviceSynchronize();

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	InitPerPixel<<<numBlocks, threadsPerBlock>>>(width, height, randState);
	cudaDeviceSynchronize();
}
