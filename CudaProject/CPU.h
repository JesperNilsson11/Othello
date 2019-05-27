#pragma once
#include "Board.h"
#include "Timer.h"

extern std::ofstream of;
extern std::ofstream pos;
extern std::ofstream lines;

int nr = 0;
int nra = 0;

struct Node {
	std::vector<Node*> children;
	int score;
	bool pruned = false;

	Node() = default;
	~Node() {
		for (auto p : children)
			delete p;
	}
};

int calculateScore(const Board& b) {
	int P = 0;
	int O = 0;

	for (int x = 0; x < 8; ++x)
		for (int y = 0; y < 8; ++y) {
			if (b.board[y][x] == 'P')
				++P;
			else if (b.board[y][x] == 'O')
				++O;
		}

	return P - O;
}

void buildTree(const Board& b, Node* n, int level) {
	nr++;
	if (level > 0) {
		std::vector<PosMov> moves;
		bool playersTurn = level % 2 == 0;
		b.updateMoves(playersTurn, &moves);

		if (moves.size() == 0) {
			n->children.push_back(new Node);
			buildTree(b, n->children[0], level - 1);
		}
		else {
			if (playersTurn) {
				for (unsigned int i = 0; i < moves.size(); ++i) {
					Board nb(b);
					Node* nn = new Node;
					n->children.push_back(nn);
					nb.movePlayer(moves[i].move, moves);
					buildTree(nb, nn, level - 1);
				}
			}
			else {
				for (unsigned int i = 0; i < moves.size(); ++i) {
					Board nb(b);
					Node* nn = new Node;
					n->children.push_back(nn);
					nb.moveOpponent(moves[i].move, moves);
					buildTree(nb, nn, level - 1);
				}
			}
		}

		int m = n->children[0]->score;
		if (playersTurn) {
			for (auto c : n->children)
				if (c->score > m)
					m = c->score;
		}
		else {
			for (auto c : n->children)
				if (c->score < m)
					m = c->score;
		}
		n->score = m;
	}
	else {
		n->score = calculateScore(b);
	}
}

void alphaBeta(const Board& b, Node* n, int level, int max, int min) {
	nra++;
	if (level > 0) {
		std::vector<PosMov> moves;
		bool playersTurn = level % 2 == 0;
		b.updateMoves(playersTurn, &moves);

		if (moves.size() == 0) {
			n->children.push_back(new Node);
			alphaBeta(b, n->children[0], level - 1, max, min);
		}
		else {
			bool prune = n->pruned;
			if (playersTurn) {
				int value = -10000000;
				for (unsigned int i = 0; i < moves.size(); ++i) {
					Board nb(b);
					Node* nn = new Node;
					nn->pruned = prune;
					n->children.push_back(nn);
					nb.movePlayer(moves[i].move, moves);
					alphaBeta(nb, nn, level - 1, max, min);
					if (value < n->children[i]->score)
						value = n->children[i]->score;
					if (max < value)
						max = value;

					if (max >= min) {
						prune = true;
						//break;
					}
				}
			}
			else {
				int value = 10000000;
				for (unsigned int i = 0; i < moves.size(); ++i) {
					Board nb(b);
					Node* nn = new Node;
					nn->pruned = prune;
					n->children.push_back(nn);
					nb.moveOpponent(moves[i].move, moves);
					alphaBeta(nb, nn, level - 1, max, min);
					if (value > n->children[i]->score)
						value = n->children[i]->score;
					if (min > value)
						min = value;

					if (max >= min) {
						prune = true;
						//break;
					}
				}
			}
		}

		int m = n->children[0]->score;
		if (playersTurn) {
			for (auto c : n->children)
				if (c->pruned == false && c->score > m)
					m = c->score;
		}
		else {
			for (auto c : n->children)
				if (c->pruned == false && c->score < m)
					m = c->score;
		}
		n->score = m;
	}
	else {
		n->score = calculateScore(b);
	}
}

void printTree(Node* n, int level) {
	if (level == 1)
		of << n->score << " ";
	else if (level > 1) {
		//if (level == 2)of << n->children.size() << " ";
		for (auto c : n->children) {
			printTree(c, level - 1);
		}
		of << "\t";
	}
}

void printDot(Node* n, float left, float right, float y) {
	constexpr float step = 10;
	float x = (right - left) / 2 + left;
	pos << x << " " << y << " " << n->pruned << "\n";
	float width = (right - left) / n->children.size();
	float start = left + width / 2;
	for (int i = 0; i < n->children.size(); ++i) {
		float lborder = left + i * width;
		lines << x << " " << y << " " << lborder + width / 2 << " " << y-step << "\n";
		printDot(n->children[i], lborder, left + (i + 1) * width, y - step);
		//data << start + i * width << " " << y - 10 << "\n";
	}
}

int AIMove(const Board& b) {

	of << "AI test ";
	Timer t;
	Node head;
	Node alphaHead;
	std::vector<PosMov> moves;
	b.updateMoves(false, &moves);

	if (moves.size() == 0) {
		head.children.push_back(new Node);
		buildTree(b, head.children[0], 4);

		return -1;
	}
	else {
		for (unsigned int i = 0; i < moves.size(); ++i) {
			Board nb(b);
			Node* n = new Node;
			head.children.push_back(n);
			nb.moveOpponent((int)moves[i].move, moves);
			buildTree(nb, n, 4);
			Node* alphan = new Node;
			alphaHead.children.push_back(alphan);
			alphaBeta(nb, alphan, 4, -1000000, 10000000);
		}
	}
	unsigned int index = 0;
	head.score = head.children[0]->score;
	for (unsigned int i = 1; i < moves.size(); ++i)
		if (head.score > head.children[i]->score) {
			index = i;
			head.score = head.children[i]->score;
		}

	unsigned int aindex = 0;
	alphaHead.score = alphaHead.children[0]->score;
	for (unsigned int i = 1; i < moves.size(); ++i)
		if (alphaHead.score > alphaHead.children[i]->score) {
			aindex = i;
			alphaHead.score = alphaHead.children[i]->score;
		}

	
	/*of << head.score << "\n";
	std::vector<Node*> stack;
	for (auto c : head.children) {
		stack.push_back(c);
		of << c->score << " ";
	}*/
	//of << "\n";
	//for (int i = 1; i <= 6; ++i) {
	//	printTree(&head, i);
	//	of << std::endl;
	//}
	//
	//printDot(&head, -200, 200, 100);
	printDot(&alphaHead, -200, 200, 100);
	
	of << nr << " " << nra << std::endl;
	nr = 0;
	nra = 0;

	return (int)moves[index].move;
}