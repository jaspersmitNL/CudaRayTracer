#include <cstdio>
#include <ctime>
#include <gl/glew.h>
#include <GLFW/glfw3.h>

#include "core/app.hpp"
#include "Render.cuh"
#include <curand.h>
#include <curand_kernel.h>



struct Vertex
{
	float x, y;
	float u, v;
};

void draw(Vertex* vertices, int numVertices)
{
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < numVertices; i++)
	{
		Vertex vertex = vertices[i];
		glTexCoord2f(vertex.u, vertex.v);
		glVertex2f(vertex.x, vertex.y);
	}
	glEnd();
}


Camera* camera;
curandState* randState;

class RayTracer : public App
{
private:
	uint32_t* m_pixels;
	uint32_t m_texture;
	Vertex* vertices;

public:
	void OnInit() override
	{
		printf("RayTracer::OnInit()\n");

		int width = GetWindowSize().x;
		int height = GetWindowSize().y;

		cudaMallocManaged(&m_pixels, width * height * sizeof(uint32_t));
		cudaMallocManaged(&camera, sizeof(Camera));
		cudaMallocManaged(&randState, width * height * sizeof(curandState));


		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_pixels);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glGenerateMipmap(GL_TEXTURE_2D);


		vertices = new Vertex[6];
		vertices[0] = {-1.0, 1.0, 0.0, 1.0}; // top left
		vertices[1] = {-1.0, -1.0, 0.0, 0.0}; // bottom left
		vertices[2] = {1.0, -1.0, 1.0, 0.0}; // bottom right
		vertices[3] = {1.0, -1.0, 1.0, 0.0}; // bottom right
		vertices[4] = {1.0, 1.0, 1.0, 1.0}; // top right
		vertices[5] = {-1.0, 1.0, 0.0, 1.0}; // top left


		Init(width, height, randState);
	}

	void OnUpdate() override
	{
	}

	void OnRender() override
	{
		int width = GetWindowSize().x;
		int height = GetWindowSize().y;



		Render(GetWindowSize(), m_pixels, camera, randState);


		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, m_pixels);
		glEnable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0);
		draw(vertices, 6);
	}

	void OnResize(glm::vec2 size) override
	{
		glViewport(0, 0, size.x, size.y);
	}

	void OnKeyPress(int key) override
	{
		glm::vec3 camPos;
		cudaMemcpy(&camPos, &camera->position, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		float speed = 0.02f;
		if (key == GLFW_KEY_W)
		{
			camPos.z -= speed;
		}
		else if (key == GLFW_KEY_S)
		{
			camPos.z += speed;
		}
		else if (key == GLFW_KEY_A)
		{
			camPos.x -= speed;
		}
		else if (key == GLFW_KEY_D)
		{
			camPos.x += speed;
		}

		cudaMemcpy(&camera->position, &camPos, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		
	}

	void OnExit() override
	{
		printf("Exit...\n");
		cudaFree(&m_pixels);
	}
};


int main()
{
	App::m_instance = new RayTracer();
	App::GetInstance()->Run("RayTracer", {800, 800});
}
