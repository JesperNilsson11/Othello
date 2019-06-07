#include "Board.h"
#include "Timer.h"

bool Board::testMove(int x, int y, int dx, int dy, char player, char opponent) const {
	//bool possible = false;
	int nx = x + dx;
	int ny = y + dy;

	if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8) {
		if (board[ny][nx] != opponent) {
			return false;
		}
		else {
			nx += dx;
			ny += dy;
			while (nx >= 0 && nx < 8 && ny >= 0 && ny < 8) {
				if (board[ny][nx] == player) {
					return true;
				}
				else if (board[ny][nx] == ' ') {
					break;
				}

				nx += dx;
				ny += dy;
			}
		}
	}

	return false;
}

bool Board::possibleMove(PosMov* pm, bool p) const {
	bool pos = false;
	int x = pm->move % 8;
	int y = pm->move / 8;
	if (board[y][x] != ' ')
		return false;

	int index = 0;
	char player = p ? 'P' : 'O';
	char opponent = !p ? 'P' : 'O';

	pm->moves[index] = testMove(x, y, -1, -1, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;
	pm->moves[index] = testMove(x, y, 0, -1, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;
	pm->moves[index] = testMove(x, y, 1, -1, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;

	pm->moves[index] = testMove(x, y, -1, 0, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;
	pm->moves[index] = testMove(x, y, 1, 0, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;

	pm->moves[index] = testMove(x, y, -1, 1, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;
	pm->moves[index] = testMove(x, y, 0, 1, player, opponent);
	if (pm->moves[index])
		pos = true;
	index++;
	pm->moves[index] = testMove(x, y, 1, 1, player, opponent);
	if (pm->moves[index])
		pos = true;

	return pos;
}

void Board::updateMoves(bool p, std::vector<PosMov>* moves) const {
	//std::cout << "cpu update moves\n";
	//Timer timer;
	PosMov pm;
	moves->clear();

	for (char i = 0; i < 64; ++i) {
		pm.move = i;
		if (possibleMove(&pm, p)) {
			moves->push_back(pm);
		}
	}
}

void Board::moveDir(int x, int y, int dx, int dy, char player) {
	x = x + dx;
	y = y + dy;

	while (board[y][x] != player) {
#if _DEBUG
		if (y < 0 || y > 7 || x < 0 || x > 7) {
			throw "\nBuggy Code";
		}
		if (board[y][x] == ' ')
			throw "\nBuggy Code";
#endif
		board[y][x] = player;
		x += dx;
		y += dy;
	}
}

bool Board::movePlayer(int i, const std::vector<PosMov>& moves) {
	for (auto& pm : moves) {
		if (pm.move == i) {
			int x = i % 8;
			int y = i / 8;

			constexpr char c = 'P';
			board[y][x] = c;
			if (pm.moves[0])
				moveDir(x, y, -1, -1, c);
			if (pm.moves[1])
				moveDir(x, y, 0, -1, c);
			if (pm.moves[2])
				moveDir(x, y, 1, -1, c);

			if (pm.moves[3])
				moveDir(x, y, -1, 0, c);
			if (pm.moves[4])
				moveDir(x, y, 1, 0, c);

			if (pm.moves[5])
				moveDir(x, y, -1, 1, c);
			if (pm.moves[6])
				moveDir(x, y, 0, 1, c);
			if (pm.moves[7])
				moveDir(x, y, 1, 1, c);

			return true;
		}
	}

	return false;
}

bool Board::moveOpponent(int i, const std::vector<PosMov>& moves) {
	for (auto& pm : moves) {
		if (pm.move == i) {
			int x = i % 8;
			int y = i / 8;

			constexpr char c = 'O';
			board[y][x] = c;
			if (pm.moves[0])
				moveDir(x, y, -1, -1, c);
			if (pm.moves[1])
				moveDir(x, y, 0, -1, c);
			if (pm.moves[2])
				moveDir(x, y, 1, -1, c);

			if (pm.moves[3])
				moveDir(x, y, -1, 0, c);
			if (pm.moves[4])
				moveDir(x, y, 1, 0, c);

			if (pm.moves[5])
				moveDir(x, y, -1, 1, c);
			if (pm.moves[6])
				moveDir(x, y, 0, 1, c);
			if (pm.moves[7])
				moveDir(x, y, 1, 1, c);

			return true;
		}
	}

	return false;
}