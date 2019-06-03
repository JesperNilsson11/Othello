#include <windows.h>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>

using namespace std;

// maximum mumber of lines the output console should have
static const WORD MAX_CONSOLE_LINES = 500;
void RedirectIOToConsole()
{
	std::wcout.clear();
	std::cout.clear();
	std::wcerr.clear();
	std::cerr.clear();
	std::wcin.clear();
	std::cin.clear();
	int hConHandle;
	long lStdHandle;
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	FILE *fp;
	bool result = true;

	// allocate a console for this app
	AllocConsole();
	if (GetStdHandle(STD_INPUT_HANDLE) != INVALID_HANDLE_VALUE)
		if (freopen_s(&fp, "CONIN$", "r", stdin) != 0)
			result = false;
		else
			setvbuf(stdin, NULL, _IONBF, 0);

	// Redirect STDOUT if the console has an output handle
	if (GetStdHandle(STD_OUTPUT_HANDLE) != INVALID_HANDLE_VALUE)
		if (freopen_s(&fp, "CONOUT$", "w", stdout) != 0)
			result = false;
		else
			setvbuf(stdout, NULL, _IONBF, 0);

	// Redirect STDERR if the console has an error handle
	if (GetStdHandle(STD_ERROR_HANDLE) != INVALID_HANDLE_VALUE)
		if (freopen_s(&fp, "CONOUT$", "w", stderr) != 0)
			result = false;
		else
			setvbuf(stderr, NULL, _IONBF, 0);
	// set the screen buffer to be big enough to let us scroll text
	/*GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
	coninfo.dwSize.Y = MAX_CONSOLE_LINES;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);

	// redirect unbuffered STDOUT to the console
	lStdHandle = (long)GetStdHandle(STD_OUTPUT_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen(hConHandle, "w");
	*stdout = *fp;
	setvbuf(stdout, NULL, _IONBF, 0);

	// redirect unbuffered STDIN to the console
	lStdHandle = (long)GetStdHandle(STD_INPUT_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen(hConHandle, "r");
	*stdin = *fp;
	setvbuf(stdin, NULL, _IONBF, 0);

	// redirect unbuffered STDERR to the console
	lStdHandle = (long)GetStdHandle(STD_ERROR_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen(hConHandle, "w");
	*stderr = *fp;
	setvbuf(stderr, NULL, _IONBF, 0);
	*/
	// make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog
	// point to console as well
	ios::sync_with_stdio();

	std::wcout.clear();
	std::cout.clear();
	std::wcerr.clear();
	std::cerr.clear();
	std::wcin.clear();
	std::cin.clear();
}