#pragma once
#include <glm/glm.hpp>
struct GLFWwindow;


class App
{
private:
	glm::vec2 m_windowSize;

public:
	GLFWwindow* m_window;

	void Run(const char* name, glm::vec2 size);
	glm::vec2 GetWindowSize();

	virtual void OnInit() = 0;
	virtual void OnUpdate() = 0;
	virtual void OnRender() = 0;
	virtual void OnResize(glm::vec2 size) = 0;
	virtual void OnKeyPress(int key) =0;
	virtual void OnExit() = 0;

	static App* GetInstance()
	{
		return m_instance;
	}

	static App* m_instance;
};
