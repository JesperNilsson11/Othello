#pragma once
#include <Windows.h>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

class Window {
public:
	Window(HINSTANCE hins, int nCmdShow);

	void MSGLoop();
private:
	HWND hwnd;
};