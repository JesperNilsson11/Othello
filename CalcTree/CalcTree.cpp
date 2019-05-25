// CalcTree.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

struct Node {
	int score;
	std::vector<Node*> children;

	Node() = default;
	~Node() {
		for (auto p : children)
			delete p;
	}
};

int main()
{
	ifstream inf;
	string line;
	inf.open("../CudaProject/console.txt");
	getline(inf, line);
	getline(inf, line);
	Node head;
	Node* current = &head;
	getline(inf, line);
	istringstream iss(line);
	int score;
	iss >> score;
	head.score = score;

	for (int i = 2; i <= 6; ++i) {
		getline(inf, line);
		istringstream iss(line);
		int index = 0;
		
		int nr;
		while (iss >> nr) {
			for (int j = 0; j < nr; ++j) {
				current->children.push_back(new Node);
				iss >> score;
				current->children[j]->score = score;
			}
		}
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
