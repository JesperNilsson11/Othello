#include "Window.h"
#include "Header.h"
#include <string>
#include "Board.h"
#include "CPU.h"

//#define PorO(p) ((p) ? 'P': 'O')

Board b = {};
bool p_turn = true;
std::vector<PosMov> moves;
/*char board[BOARDSIZE][BOARDSIZE] = {
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', 'O', 'P', ' ', ' ', ' ',
		' ', ' ', ' ', 'P', 'O', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
};*/

HBRUSH red = CreateSolidBrush(RGB(255, 0, 0));
HBRUSH green = CreateSolidBrush(RGB(0, 255, 0));
HBRUSH blue = CreateSolidBrush(RGB(0, 0, 255));
HBRUSH white = CreateSolidBrush(RGB(255, 255, 255));
HBRUSH black = CreateSolidBrush(RGB(0, 0, 0));
//int i = 0;

bool gameOver() {
	for (int y = 0; y < BOARDSIZE; ++y) {
		for (int x = 0; x < BOARDSIZE; ++x) {
			if (b.board[y][x] != 'P' && b.board[y][x] != 'O')
				return false;
		}
	}

	return true;
}

void drawBoard(HWND hwnd) {
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hwnd, &ps);
	RECT rect;
	rect.top = 0;
	rect.bottom = BOARDSIZE * TILESIZE;
	rect.left = 0;
	rect.right = BOARDSIZE * TILESIZE;
	FillRect(hdc, &rect, black);
	for (int y = 0; y < BOARDSIZE; ++y)
		for (int x = 0; x < BOARDSIZE; ++x) {
			rect.top = TILESIZE * y;
			rect.bottom = TILESIZE * (y + 1) - 1;
			rect.left = TILESIZE * x;
			rect.right = TILESIZE * (x + 1) - 1;
			FillRect(hdc, &rect, green);
			if (b.board[y][x] == 'P') {
				SelectObject(hdc, black);
				Ellipse(hdc, rect.left + 1, rect.top + 1, rect.right - 1, rect.bottom - 1);
				//FillRect(hdc, &rect, black);//(board[y][x] == 1 ? green : blue)
			}
			else if (b.board[y][x] == 'O') {
				SelectObject(hdc, white);
				Ellipse(hdc, rect.left + 1, rect.top + 1, rect.right - 1, rect.bottom - 1);
				//FillRect(hdc, &rect, white);
			}
			//else
				//FillRect(hdc, &rect, green);
		}

	//FillRect(hdc, &r, green);//ps.rcPaint
	EndPaint(hwnd, &ps);
	//SetWindowTextA(hwnd, (std::to_string(board[0][0])).c_str());
}

void gameLogic(HWND hwnd, int move) {
	//int aiMove;
	if (p_turn) {
		if (b.movePlayer(move, moves)) {
			p_turn = !p_turn;
			b.updateMoves(p_turn, &moves);
			InvalidateRect(hwnd, 0, TRUE);
			drawBoard(hwnd);
			//aiMove = AIMove(b);
			//Sleep(100);
			PostMessage(hwnd, WM_USER, 0, 0);
			/*// TEST
			if (b.moveOpponent(aiMove, moves)) {
				p_turn = !p_turn;
				b.updateMoves(p_turn, &moves);
			}*/
		}
	}
	else {
		if (b.moveOpponent(move, moves)) {
			p_turn = !p_turn;
			b.updateMoves(p_turn, &moves);
			InvalidateRect(hwnd, 0, TRUE);
			drawBoard(hwnd);
		}
	}

	if (moves.size() == 0) {
		p_turn = !p_turn;
		b.updateMoves(p_turn, &moves);
	}

	if (gameOver())
		SetWindowTextA(hwnd, (std::string("Reversi ") + std::string((calculateScore(b) > 0 ? "Player won" : std::string("Opponent won")))).c_str());
	else
		SetWindowTextA(hwnd, (std::string("Reversi ") + std::string((p_turn ? "Players turn" : std::string("Opponents turn")/* + std::to_string(move)*/))).c_str());
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;

	case WM_PAINT:
	{
		drawBoard(hwnd);
		return 0;
	}
	case WM_LBUTTONDOWN: {
		POINT p;
		GetCursorPos(&p);
		ScreenToClient(hwnd, &p);
		int x = p.x / TILESIZE;
		int y = p.y / TILESIZE;
		gameLogic(hwnd, x + y * BOARDSIZE);
		
		//SetWindowTextA(hwnd, (std::to_string(x) + std::string(" ") + std::to_string(y)).c_str());
		//hb = CreateSolidBrush(RGB(255, 0, 0));
		return 0;
	}
	case WM_CLOSE:
		DeleteObject(red);
		DeleteObject(green);
		DeleteObject(blue);
		DeleteObject(white);
		DeleteObject(black);
		DestroyWindow(hwnd);
		return 0;
	case WM_USER:
		Sleep(1000);
		int aiMove = AIMove(b);
		gameLogic(hwnd, aiMove);
		//SetWindowTextA(hwnd, "Its Working");
		return 0;
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

Window::Window(HINSTANCE hins, int nCmdShow) {
	b.updateMoves(p_turn, &moves);

	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hins;
	wc.lpszClassName = "Reversi";
	RegisterClass(&wc);

	DWORD style = WS_BORDER | WS_CAPTION | WS_SYSMENU;

	RECT r;
	r.bottom = BOARDSIZE * TILESIZE;
	r.top = 0;
	r.left = 0;
	r.right = BOARDSIZE * TILESIZE;
	if (!AdjustWindowRect(&r, style, FALSE))
		throw "HAhs";

	/*const int cx = GetSystemMetrics(SM_CXBORDER);
	const int cy = GetSystemMetrics(SM_CYBORDER);
	r.right += cx *2;
	r.bottom += cy*2;
	*/
	hwnd = CreateWindowA(
		//0,                              // Optional window styles.
		"Reversi",                     // Window class
		"Reversi",						// Window text
		style,            // Window style
		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, r.right-r.left, r.bottom-r.top,
		NULL,       // Parent window    
		NULL,       // Menu
		hins,  // Instance handle
		NULL        // Additional application data
	);

	if (hwnd == NULL)
		throw "No window";

	ShowWindow(hwnd, nCmdShow);
	RECT rc;
	GetClientRect(hwnd, &rc);
	//SetWindowTextA(hwnd, (std::to_string(rc.right- rc.left) + std::string(" ") + std::to_string(rc.bottom-rc.top)).c_str());
}

void Window::MSGLoop() {
	MSG msg = {};
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
