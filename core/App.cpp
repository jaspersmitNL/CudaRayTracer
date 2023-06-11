#include "app.hpp"
#include <gl/glew.h>
#include <GLFW/glfw3.h>


void App::Run(const char* name, glm::vec2 size)
{
	glfwInit();
	glfwDefaultWindowHints();
	m_windowSize = size;
	this->m_window = glfwCreateWindow(m_windowSize.x, m_windowSize.y, name, 0, 0);
	glfwMakeContextCurrent(m_window);
	glewInit();

	glfwSetWindowSizeCallback(m_window, [](GLFWwindow*, int w, int h)
	{
		glm::vec2 windowSize(w, h);
		m_instance->m_windowSize = windowSize;
		m_instance->OnResize(windowSize);
	});

	glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			m_instance->OnKeyPress(key);
		}
	});

	OnInit();

	while (!glfwWindowShouldClose(m_window))
	{
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		OnUpdate();
		OnRender();


		glfwSwapBuffers(m_window);
	}
	OnExit();
}

glm::vec2 App::GetWindowSize()
{
	return m_windowSize;
}


App* App::m_instance = nullptr;
