#pragma once
#include <vector>
#include "Header.h"

struct PosMov {
	char move = -1;
	bool moves[8] = { false,false,false,false,false,false,false,false };
};

struct Board {
	char board[BOARDSIZE][BOARDSIZE] = {
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', 'O', 'P', ' ', ' ', ' ',
		' ', ' ', ' ', 'P', 'O', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
	};
	Board() = default;
	Board(const Board& b) {
		for (int x = 0; x < 8; ++x)
			for (int y = 0; y < 8; ++y)
				board[y][x] = b.board[y][x];
	}
	Board& operator=(const Board& r) {
		for (int x = 0; x < 8; ++x)
			for (int y = 0; y < 8; ++y)
				board[y][x] = r.board[y][x];

		return *this;
	}

	bool testMove(int x, int y, int dx, int dy, char player, char opponent) const;
	bool possibleMove(PosMov* pm, bool X) const;
	void updateMoves(bool p, std::vector<PosMov>* moves) const;
	void moveDir(int x, int y, int dx, int dy, char player);
	bool movePlayer(int i, const std::vector<PosMov>& moves);
	bool moveOpponent(int i, const std::vector<PosMov>& moves);
};