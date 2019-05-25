#pragma once
#include <chrono>
#include <fstream>

extern std::ofstream of;

struct Timer {
	Timer() {
		start = std::chrono::steady_clock::now();
	}
	~Timer() {
		std::chrono::duration<float> dur = std::chrono::steady_clock::now() - start;
		of << "Time " << (dur.count() *1000.f) << "ms" << std::endl;
	}

private:
	std::chrono::time_point<std::chrono::steady_clock> start;
};
