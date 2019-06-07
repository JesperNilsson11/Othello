#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include "Window.h"
#include "console.h"
#include "Timer.h"

using namespace std;

std::ofstream of;
std::ofstream pos;
std::ofstream lines;
std::ofstream times;
std::ofstream nodes;

#define CUDA(call, string) cudaStatus = (call); if (cudaStatus != cudaSuccess) {cout << string << " " << cudaGetErrorString(cudaStatus) << endl; return cudaStatus;}
#define CUDAE(call) CUDA(call, "Error")
#define CUDANORETURN(call, string) cudaStatus = (call); if (cudaStatus != cudaSuccess) {cout << string << " " << cudaGetErrorString(cudaStatus) << endl; }
#define CHECK(call) CUDANORETURN(call, "Error")

cudaError_t callCuda(int* data, int count);

int profileCopies(float        *h_a,
	float        *h_b,
	float        *d,
	unsigned int  n,
	char         *desc)
{
	cudaError_t cudaStatus;
	cout << "\n" << desc <<  " transfers\n";

	unsigned int bytes = n * sizeof(float);

	// events for timing
	cudaEvent_t startEvent, stopEvent;

	CUDAE(cudaEventCreate(&startEvent));
	CUDAE(cudaEventCreate(&stopEvent));

	CUDAE(cudaEventRecord(startEvent, 0));
	CUDAE(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
	CUDAE(cudaEventRecord(stopEvent, 0));
	CUDAE(cudaEventSynchronize(stopEvent));

	float time;
	CUDAE(cudaEventElapsedTime(&time, startEvent, stopEvent));
	cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-6 / time << " time: " << time << "\n";

	CUDAE(cudaEventRecord(startEvent, 0));
	CUDAE(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
	CUDAE(cudaEventRecord(stopEvent, 0));
	CUDAE(cudaEventSynchronize(stopEvent));

	CUDAE(cudaEventElapsedTime(&time, startEvent, stopEvent));
	cout << "  Device to Host bandwidth (GB/s): " << bytes * 1e-6 / time << "\n";

	for (int i = 0; i < n; ++i) {
		if (h_a[i] != h_b[i]) {
			cout << "*** " << desc << "transfers failed ***\n";
			break;
		}
	}

	// clean up events
	CUDAE(cudaEventDestroy(startEvent));
	CUDAE(cudaEventDestroy(stopEvent));

	return 0;
}

__device__ bool testMove(char* board, char x, char y, char dx, char dy, char player, char oppo) {
	x += dx;
	y += dy;
	
	if (x >= 0 && x < 8 && y >= 0 && y < 8) {
		if (board[y * 8 + x] != oppo) {
			return false;
		}
		else {
			x += dx;
			y += dy;
			while (x >= 0 && x < 8 && y >= 0 && y < 8) {
				if (board[y * 8 + x] == player) {
					return true;
				}
				else if (board[y * 8 + x] == ' ') {
					break;
				}
	
				x += dx;
				y += dy;
			}
		}
	}
	
	return false;
}

__device__ struct Move {
	bool dir[8];
	char move;
};

__device__ struct Moves {
	Move moves[30];
	int nr;
};

__device__ Moves cudaMoves;
__device__ void updateMoves(char* board, Moves* moves, char player) {
	__shared__ Move s_moves[64];
	//__shared__ char s_results[9 * 64];
	__shared__ bool s_bools[64];
	__shared__ char s_board[64];
	//int index = threadIdx.y * 8 + threadIdx.x;
	if (threadIdx.x == 0) {
		s_moves[threadIdx.y].move = threadIdx.y;
		s_bools[threadIdx.y] = false;
		//s_results[threadIdx.y * 9] = threadIdx.y;
		s_board[threadIdx.y] = board[threadIdx.y];
	}
	//if (blockIdx.x == 0) {
	//	//s_bools[0] = 1;
	//	s_results[0] = 1;
	//	s_results[1] = 2;
	//	s_results[2] = 3;
	//	s_results[3] = 4;
	//	s_results[4] = 5;
	//	s_results[5] = 6;
	//	s_results[6] = 7;
	//	s_results[9] = 8;
	//	//s_results[9+9] = 9;
	//}

	//if (threadIdx.x == 0 && threadIdx.y == 0)
	//	*(int*)results = 0;
	__syncthreads();// Not needed cuz sync further down?

	char x = threadIdx.y % 8;
	char y = threadIdx.y / 8;
	char dx;
	char dy;
	if (s_board[x + y * 8] == ' ') {
		if (threadIdx.x == 0 || threadIdx.x == 3 || threadIdx.x == 5)
			dx = -1;
		else if (threadIdx.x == 1 || threadIdx.x == 6)
			dx = 0;
		else
			dx = 1;

		if (threadIdx.x < 3)
			dy = -1;
		else if (threadIdx.x > 4)
			dy = 1;
		else
			dy = 0;


		//int index = threadIdx.y * 9 + threadIdx.x + 1;
		char oppo = 'O';
		if (player == 'O')
			oppo = 'P';
		s_moves[threadIdx.y].dir[threadIdx.x] = testMove(s_board, x, y, dx, dy, player, oppo);
		if (s_moves[threadIdx.y].dir[threadIdx.x]) {
			s_bools[threadIdx.y] = true;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		//int id = atomicAdd((int*)(&moves->nr), 1);
		if (s_bools[threadIdx.y]) {
			int id = atomicAdd((int*)(&moves->nr), 1);

			//DEBUG CODE
			if (id >= 30)
				printf("To many moves!!!!!!!!!%d\n", id);
			//for (int i = 0; i < 9; ++i) {
			//	results[id * 9 + 4 + i] = s_results[threadIdx.y * 9 + i];
			//}

			moves->moves[id] = s_moves[threadIdx.y];
		}
	}
	//moves->nr = 11;
	// testing if working *(int*)results = 11;
}
__global__ void updateMovesKernel(char* board, Moves* moves, char player) {
	__shared__ Move s_moves[64];
	//__shared__ char s_results[9 * 64];
	__shared__ bool s_bools[64];
	__shared__ char s_board[64];

	//int index = threadIdx.y * 8 + threadIdx.x;
	if (threadIdx.x == 0) {
		s_moves[threadIdx.y].move = threadIdx.y;
		s_bools[threadIdx.y] = false;
		//s_results[threadIdx.y * 9] = threadIdx.y;
		
		s_board[threadIdx.y] = board[threadIdx.y];

	}
	//if (blockIdx.x == 0) {
	//	//s_bools[0] = 1;
	//	s_results[0] = 1;
	//	s_results[1] = 2;
	//	s_results[2] = 3;
	//	s_results[3] = 4;
	//	s_results[4] = 5;
	//	s_results[5] = 6;
	//	s_results[6] = 7;
	//	s_results[9] = 8;
	//	//s_results[9+9] = 9;
	//}

	//if (threadIdx.x == 0 && threadIdx.y == 0)
	//	*(int*)results = 0;
	__syncthreads();// Not needed cuz sync further down?

	char x = threadIdx.y % 8;
	char y = threadIdx.y / 8;
	char dx;
	char dy;
	if (s_board[x + y * 8] == ' ') {
		if (threadIdx.x == 0 || threadIdx.x == 3 || threadIdx.x == 5)
			dx = -1;
		else if (threadIdx.x == 1 || threadIdx.x == 6)
			dx = 0;
		else
			dx = 1;

		if (threadIdx.x < 3)
			dy = -1;
		else if (threadIdx.x > 4)
			dy = 1;
		else
			dy = 0;


		//int index = threadIdx.y * 9 + threadIdx.x + 1;
		char oppo = 'O';
		if (player == 'O')
			oppo = 'P';
		s_moves[threadIdx.y].dir[threadIdx.x] = testMove(s_board, x, y, dx, dy, player, oppo);
		if (s_moves[threadIdx.y].dir[threadIdx.x]) {
			s_bools[threadIdx.y] = true;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		//int id = atomicAdd((int*)(&moves->nr), 1);
		if (s_bools[threadIdx.y]) {
			int id = atomicAdd((int*)(&moves->nr), 1);

			//DEBUG CODE
			if (id >= 30)
				printf("To many moves!!!!!!!!!%d\n", id);
			//for (int i = 0; i < 9; ++i) {
			//	results[id * 9 + 4 + i] = s_results[threadIdx.y * 9 + i];
			//}

			moves->moves[id] = s_moves[threadIdx.y];
		}
	}
	//moves->nr = 11;
	// testing if working *(int*)results = 11;
}

__device__ struct CUDANode {
	CUDANode* children[30];
	int score;
};

__device__ int calculateScore(char* b) {
	int P = 0;
	int O = 0;

	for (int x = 0; x < 64; ++x)
		if (b[x] == 'P')
			++P;
		else if (b[x] == 'O')
			++O;

	return P - O;
}

//__device__ int d_write;

__device__ void moveDir(char* board, int x, int y, int dx, int dy, char player) {
	//int temp = x + y * 8;
	//char tempB[64];
	//for (int i = 0; i < 64; ++i)
	//	tempB[i] = board[i];
	x = x + dx;
	y = y + dy;

	while (board[y* 8 + x] != player) {
		board[y*8+x] = player;
		x += dx;
		y += dy;

		if (y < 0 || y > 7 || x < 0 || x > 7) {
			//int id = atomicAdd(&d_write, 1);
			//if (id == 0) {

			printf("Buggy cuda\n"/*PLayer: %c\nlevel: %d index: %d dx: %d dy: %d\n", player, level, temp, dx, dy*/);
			//for (int i = 0; i < 64; ++i) {
			//	printf("%d%c ", i, tempB[i]);
			//	if (i % 8 == 7)
			//		printf("\n");
			//}
			//printf("===========\n");
			//for (int i = 0; i < 64; ++i) {
			//	printf("%d%c ", i, board[i]);
			//	if (i % 8 == 7)
			//		printf("\n");
			//}
			//d_write = 0;
			//}

			return;
		}
	}
}

__device__ void move(char* board, int i, char c, Move* move) {
	int x = i % 8;
	int y = i / 8;

	//printf("move %d\n", i);
	if (move->dir[0])
		moveDir(board, x, y, -1, -1, c);
	if (move->dir[1])
		moveDir(board, x, y, 0, -1, c);
	if (move->dir[2])
		moveDir(board, x, y, 1, -1, c);

	if (move->dir[3])
		moveDir(board, x, y, -1, 0, c);
	if (move->dir[4])
		moveDir(board, x, y, 1, 0, c);

	if (move->dir[5])
		moveDir(board, x, y, -1, 1, c);
	if (move->dir[6])
		moveDir(board, x, y, 0, 1, c);
	if (move->dir[7])
		moveDir(board, x, y, 1, 1, c);
	
	//tempOrary bottom?
	board[i] = c;
}

__device__ char cudaBoard[64];
__device__ CUDANode headCuda;
__device__ int result;
__device__ int bytes;
__device__ char* mem;
__device__ void* NEW(int b) {
	//printf("gpu pointer %p\n", mem);
	int align = b % 1024;
	int temp = atomicAdd((int*)(&bytes), b + 1024 - align);
	//printf("temp %d\n", temp);
	if (temp > 2000000000)
		printf("error memory\n");

	return mem + temp;
}

// NOT CALLING MOVE KERNAL
#if 1
__global__ void getMoveKernel(char* board, CUDANode* node, int level) {
	__shared__ char* newBoards[30];
	__shared__ cudaStream_t s[30];
	__shared__ int scores[30];
	__shared__ char indices[30];
	//if (level == 5)
	//	d_write = 0;
	//printf("Start %d\n", level);
	int id = threadIdx.x + threadIdx.y * 8;
	if (level > 0) {
		//__shared__ Moves* moves;
		__shared__ Moves moves;
		__shared__ char player;
		const dim3 threads(8, 64, 1);
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			player = (level % 2 == 0) ? 'P' : 'O';

			if (level == 5)
				bytes = 0;

			//moves = (Moves*)NEW(sizeof(Moves));

			//moves = new Moves;
			//mem[0] = 'A';
			//printf("Working\n");
			moves.nr = 0;
			//printf("Working?\n");
		}
		//const dim3 threads(8, 64, 1);
		//clock_t start = clock();
		__syncthreads();
		updateMoves(board, &moves, player);
		__syncthreads();
		//clock_t stop = clock();
		//printf("%lu\n", (stop-start));
		
		//if (threadIdx.x == 0 && threadIdx.y == 0) {

		if (moves.nr == 0) {
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				newBoards[0] = (char*)NEW(64);
				//newBoards[0] = new char[64];
				memcpy(newBoards[0], board, 64);
				//printf("level %d\n", level);
				node->children[0] = (CUDANode*)NEW(sizeof(CUDANode));
				//node->children[0] = new CUDANode;

				getMoveKernel << <1, threads >> > (newBoards[0], node->children[0], level - 1);
				cudaDeviceSynchronize();
			}
			//delete[] newBoards[0];
		}
		else {
			//printf("level %d\n", level);

			
			//for (int i = 0; i < moves->nr; ++i) {
			if (id < moves.nr) {
				cudaStreamCreateWithFlags(&s[id], cudaStreamNonBlocking);
				newBoards[id] = (char*)NEW(64);
				//newBoards[i] = new char[64];
				memcpy(newBoards[id], board, 64);
				//printf("level %d i %d\n", level, i);
				node->children[id] = (CUDANode*)NEW(sizeof(CUDANode));
				//node->children[i] = new CUDANode;
				move(newBoards[id], moves.moves[id].move, player, &moves.moves[id]);


				getMoveKernel << <1, threads, 0, s[id] >> > (newBoards[id], node->children[id], level - 1);
				//getMoveKernel << <1, 1 >> > (newBoards[i], node->children[i], level - 1);
			}
			//printf("level %d\n", level);

			cudaDeviceSynchronize();
			//for (int i = 0; i < moves->nr; ++i)
			//	delete[] newBoards[i];
		}

		if (id < moves.nr) {
			scores[id] = node->children[id]->score;
			indices[id] = id;
			// doesnt need syncthreads since less than warp size
			if (player == 'P') {
				for (int step = (moves.nr + 1) / 2; id < step && step > 0; (step + 1) / 2) {
					if (id + step < moves.nr && scores[id] < scores[id + step]) {
						scores[id] = scores[id + step];
						indices[id] = indices[id + step];
					}
				}
			}
			else {
				for (int step = (moves.nr + 1) / 2; id < step && step > 0; (step + 1) / 2) {
					if (id + step < moves.nr) {
						if (scores[id] < scores[id + step]) {
							scores[id] = scores[id + step];
							indices[id] = indices[id + step];
						}
						else if (scores[id] == scores[id + step] && moves.moves[id].move < moves.moves[id + step].move)
							indices[id] = indices[id + step];
					}
				}
			}

		//int temp = node->children[0]->score;
		//int index = 0;
		//if (player == 'P') {
		//	for (int i = 1; i < moves.nr; ++i) {
		//		if (temp < node->children[i]->score) {
		//			temp = node->children[i]->score;
		//			index = i;
		//		}
		//	}
		//}
		//else {
		//	for (int i = 1; i < moves.nr; ++i) {
		//		if (temp > node->children[i]->score) {
		//			temp = node->children[i]->score;
		//			index = i;
		//		}
		//		else if (temp == node->children[i]->score && moves.moves[i].move < moves.moves[index].move)
		//			index = i;// gpu chose lowest move same as cpu
		//	}
		//}
			if (id == 0) {
				node->score = scores[0];

				if (level == 5) {
					//for (int i = 0; i < moves->nr; ++i)
					//	printf("%d ", moves->moves[i].move);

					result = moves.moves[indices[0]].move;

					//printf("\nresult cuda %d\n", result);
				}
			}
		}


		//cudaStream_t s1, s2;
		//node->left = new CUDANode;
		//cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		//getMoveKernel << <1, 1, 0,s1 >> > (node->left, level - 1);
		//cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		//node->right = new CUDANode;
		//getMoveKernel << <1, 1, 0,s2 >> > (node->right, level - 1);
		//cudaDeviceSynchronize();
		//
		//node->score = node->left->score + node->right->score;

	deletes:
		//delete node->left;
		//delete node->right;

		//for (int i = 0; i < moves->nr; ++i)
		//	delete node->children[i];
		//
		//delete moves;

		//}
		//delete[] newBoard;
	}
	else {
		//printf("leaf node\n");
		//result = 11;
		if (threadIdx.x == 0 && threadIdx.y == 0)
			node->score = calculateScore(board);
	}
	//printf("end %d\n", level);
	//char* board = new char[64];
	//for (int i = 0; i < 64; ++i)
	//	board[i] = ' ';
	//board[8 * 2 + 3] = 'P';
	//board[8 * 3 + 3] = 'P';
	//board[8 * 3 + 4] = 'P';
	//board[8 * 4 + 3] = 'P';
	//board[8 * 4 + 4] = 'O';
	//char* results = new char[4 + 9 * 10];
	//for (int i = 0; i < results[])
	//printf("number of moves: %d\n", (int)moves->nr);
	//for (int i = 0; i < moves->nr; ++i)
	//	printf("%d\n", moves->moves[i].move);
	//delete[] board;
	//delete[] results;
}
#else
__global__ void getMoveKernel(char* board, CUDANode* node, int level) {
	//if (level == 5)
	//	d_write = 0;
	//printf("Start %d\n", level);
	if (level > 0) {
		char player = (level % 2 == 0) ? 'P' : 'O';
		
		Moves* moves = new Moves;
		moves->nr = 0;
		const dim3 threads(8, 64, 1);
		//clock_t start = clock();
		updateMovesKernel << <1, threads >> > (board, moves, player);
		cudaDeviceSynchronize();
		//clock_t stop = clock();
		//printf("%lu\n", (stop-start));
		char* newBoards[20];

		if (moves->nr == 0) {
			newBoards[0] = new char[64];
			memcpy(newBoards[0], board, 64);
			node->children[0] = new CUDANode;
			
			getMoveKernel << <1, 1 >> > (newBoards[0], node->children[0], level - 1);
			cudaDeviceSynchronize();
			delete[] newBoards[0];
		}
		else {
			//printf("level %d\n", level);
			
			cudaStream_t s[20];
			for (int i = 0; i < moves->nr; ++i) {
				cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
				newBoards[i] = new char[64];
				memcpy(newBoards[i], board, 64);
				node->children[i] = new CUDANode;
				move(newBoards[i], moves->moves[i].move, player, &moves->moves[i]);


				getMoveKernel << <1, 1, 0, s[i] >> > (newBoards[i], node->children[i], level - 1);
				//getMoveKernel << <1, 1 >> > (newBoards[i], node->children[i], level - 1);
			}
			//printf("level %d\n", level);

			cudaDeviceSynchronize();
			for (int i = 0; i < moves->nr; ++i)
				delete[] newBoards[i];
		}

		int temp = node->children[0]->score;
		int index = 0;
		if (player == 'P') {
			for (int i = 1; i < moves->nr; ++i) {
				if (temp < node->children[i]->score) {
					temp = node->children[i]->score;
					index = i;
				}
			}
		}
		else {
			for (int i = 1; i < moves->nr; ++i) {
				if (temp > node->children[i]->score) {
					temp = node->children[i]->score;
					index = i;
				}
				else if (temp == node->children[i]->score && moves->moves[i].move < moves->moves[index].move)
					index = i;// gpu chose lowest move same as cpu
			}
		}
		node->score = temp;

		if (level == 5) {
			//for (int i = 0; i < moves->nr; ++i)
			//	printf("%d ", moves->moves[i].move);
			result = moves->moves[index].move;
			//printf("\nresult cuda %d\n", result);
		}
			

		//cudaStream_t s1, s2;
		//node->left = new CUDANode;
		//cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		//getMoveKernel << <1, 1, 0,s1 >> > (node->left, level - 1);
		//cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		//node->right = new CUDANode;
		//getMoveKernel << <1, 1, 0,s2 >> > (node->right, level - 1);
		//cudaDeviceSynchronize();
		//
		//node->score = node->left->score + node->right->score;

deletes:
		//delete node->left;
		//delete node->right;
		for (int i = 0; i < moves->nr; ++i)
			delete node->children[i];

		delete moves;
		//delete[] newBoard;
	}
	else {
		//printf("leaf node\n");
		//result = 11;
		node->score = calculateScore(board);
	}
	//printf("end %d\n", level);
	//char* board = new char[64];
	//for (int i = 0; i < 64; ++i)
	//	board[i] = ' ';
	//board[8 * 2 + 3] = 'P';
	//board[8 * 3 + 3] = 'P';
	//board[8 * 3 + 4] = 'P';
	//board[8 * 4 + 3] = 'P';
	//board[8 * 4 + 4] = 'O';
	//char* results = new char[4 + 9 * 10];
	//for (int i = 0; i < results[])
	//printf("number of moves: %d\n", (int)moves->nr);
	//for (int i = 0; i < moves->nr; ++i)
	//	printf("%d\n", moves->moves[i].move);
	//delete[] board;
	//delete[] results;
}
#endif

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
	RedirectIOToConsole();
	cudaError_t cudaStatus;
	//cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 6);
	//CUDANode* cudaHead = nullptr;
	//CHECK(cudaGetSymbolAddress((void**)&cudaHead, headCuda));
	//{
	//	Timer t;
	//	cudaEvent_t startEvent, stopEvent;
	//	CHECK(cudaEventCreate(&startEvent));
	//	CHECK(cudaEventCreate(&stopEvent));
	//	CHECK(cudaEventRecord(startEvent, 0));
	//	getMoveKernel << <1, 1 >> > (cudaHead, 5);
	//	CHECK(cudaEventRecord(stopEvent, 0));
	//	CHECK(cudaEventSynchronize(stopEvent));
	//
	//	float time;
	//	CHECK(cudaEventElapsedTime(&time, startEvent, stopEvent));
	//	cout << "time: " << time << "ms\n";
	//	//cudaDeviceSynchronize();
	//}
	//int score = 1;
	//void* symbol = nullptr;
	//cout << symbol;
	//CHECK(cudaGetSymbolAddress(&symbol, result));
	//cout << " " << symbol << endl;
	//CHECK(cudaMemcpy(&score, symbol, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << score << endl;
	//char* s = (char*)cudaHead;
	//s += (sizeof(CUDANode*) * 2);
	//CHECK(cudaMemcpy(&score, (void*)s, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << score << endl;
	//
	//int temp;
	//cin >> temp;
	//
	//return score;
	//parent << <1, 1 >> > ();
	//CHECK(cudaDeviceSynchronize());
	//cout << "hello" << endl;
	
	


	Window window(hInstance, nCmdShow);
	pos.open("data.txt");
	lines.open("lines.txt");
	times.open("times.txt");
	nodes.open("nodes.txt");
	of.open("console.txt", std::ios_base::app);
	of << "==============New Game==============" << std::endl;

	//cudaError_t cudaStatus;
	cout << "start\n";
	{
		const unsigned int N = 1;//1048576;
	const unsigned int bytes = N * sizeof(int);
	int *h_a = new int[N];
	int *d_a;
	cout << "end" << endl;
	cudaMalloc((int**)&d_a, bytes);

	memset(h_a, 0, bytes);
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	CUDAE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	cudaFree(d_a);
	delete[] h_a;
	}

	//example 2
	unsigned int nElements = 1 * 1024 * 1024;
	const unsigned int bytes = nElements * sizeof(float);
	// host arrays
	float *h_aPageable, *h_bPageable;
	float *h_aPinned, *h_bPinned;
	// device array
	float *d_a;
	// allocate and initialize
	h_aPageable = new float[nElements];                    // host pageable
	h_bPageable = new float[nElements];                    // host pageable
	CUDAE(cudaMallocHost((void**)&h_aPinned, bytes)); // host pinned
	CUDAE(cudaMallocHost((void**)&h_bPinned, bytes)); // host pinned
	CUDAE(cudaMalloc((void**)&d_a, bytes));           // device

	for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;

	memcpy(h_aPinned, h_aPageable, bytes);
	memset(h_bPageable, 0, bytes);
	memset(h_bPinned, 0, bytes);

	// output device info and transfer size
	cudaDeviceProp prop;
	CUDAE(cudaGetDeviceProperties(&prop, 0));

	cout << "\nDevice:" << prop.name << "\n";
	cout << "Transfer size (MB): " << bytes / (1024 * 1024) << "\n";

	// perform copies and report bandwidth
	profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
	profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

	// cleanup
	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	delete[] h_aPageable;
	delete[] h_bPageable;

	// call cuda function.
	int* data = nullptr;
	int count = 1024 * 1024;
	data = new int[count];
	cudaStatus = callCuda(data, count);
	if (cudaStatus != cudaSuccess) {
		cout << "call to Cuda failed!";
		return 1;
	}
	delete[] data;


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cout << "cudaDeviceReset failed!";
		return 1;
	}

	//parent << <1, 1 >> > ();
	cout << "end" << endl;

	void* memory;
	CHECK(cudaMalloc((void**)&memory, 2000000000));
	char* d_memory = nullptr;
	CHECK(cudaGetSymbolAddress((void**)&d_memory, mem));
	cout << "sizeof(char*) " << sizeof(char*) << endl;
	cout << "cpu pointer " << memory << endl;
	CHECK(cudaMemcpy(d_memory, &memory, sizeof(char*), cudaMemcpyHostToDevice));
	window.MSGLoop();
	of.close();
	pos.close();
	lines.close();
	times.close();
	nodes.close();

	cudaFree(memory);

	return 0;
}

cudaError_t callCuda(int* data, int count) {
	cudaError_t cudaStatus;
	int* d_a;
	int bytes = count * 4;
	CUDAE(cudaMalloc((void**)&d_a, bytes));
	CUDAE(cudaMemcpy(d_a, data, bytes, cudaMemcpyHostToDevice));
	//quicksort << <1,1 >> > (d_a, 0, count - 1);
	CUDAE(cudaMemcpy(data, d_a, bytes, cudaMemcpyDeviceToHost));
	cudaFree(d_a);

	return cudaSuccess;
}
char* getCurrentBoard();
void cudaTestMove() {
	Timer t;
	cudaError_t cudaStatus;
	//cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 6);
	char* board = nullptr;
	CHECK(cudaGetSymbolAddress((void**)&board, cudaBoard));
	CHECK(cudaMemcpy(board, getCurrentBoard(), 64, cudaMemcpyHostToDevice));

	Moves* moves = nullptr;
	CHECK(cudaGetSymbolAddress((void**)&moves, cudaMoves));
	{
		cudaEvent_t startEvent, stopEvent;
		CHECK(cudaEventCreate(&startEvent));
		CHECK(cudaEventCreate(&stopEvent));
		CHECK(cudaEventRecord(startEvent, 0));
		const dim3 threads(8, 64, 1);
		//clock_t start = clock();
		updateMovesKernel << <1, threads >> > (board, moves, 'O');
		CHECK(cudaEventRecord(stopEvent, 0));
		CHECK(cudaEventSynchronize(stopEvent));
		//std::cout << "========================" << std::endl;

		float time;
		CHECK(cudaEventElapsedTime(&time, startEvent, stopEvent));
		cout << "time update moves kernel: " << time << "ms\n";
		//cudaDeviceSynchronize();
	}
	int score = 1;
	void* symbol = nullptr;
	cout << symbol;
	CHECK(cudaGetSymbolAddress(&symbol, result));
	cout << " " << symbol << endl;
	CHECK(cudaMemcpy(&score, symbol, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << score << endl;
	//char* s = (char*)cudaHead;
	//s += (sizeof(CUDANode*) * 20);
	//int nodeScore;
	//CHECK(cudaMemcpy(&nodeScore, (void*)s, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << score << endl;
}

int cudaGetMove() {
	Timer t;
	cudaError_t cudaStatus;
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 6);
	char* board = nullptr;
	CHECK(cudaGetSymbolAddress((void**)&board, cudaBoard));
	CHECK(cudaMemcpy(board, getCurrentBoard(), 64, cudaMemcpyHostToDevice));

	CUDANode* cudaHead = nullptr;
	CHECK(cudaGetSymbolAddress((void**)&cudaHead, headCuda));
	{
		cudaEvent_t startEvent, stopEvent;
		CHECK(cudaEventCreate(&startEvent));
		CHECK(cudaEventCreate(&stopEvent));
		CHECK(cudaEventRecord(startEvent, 0));
		const dim3 threads(8, 64, 1);
		getMoveKernel << <1, threads >> > (board, cudaHead, 5);
		CHECK(cudaEventRecord(stopEvent, 0));
		CHECK(cudaEventSynchronize(stopEvent));
		std::cout << "========================" << std::endl;

		float time;
		CHECK(cudaEventElapsedTime(&time, startEvent, stopEvent));
		cout << "time: " << time << "ms\n";
		//cudaDeviceSynchronize();
	}
	int score = 1;
	void* symbol = nullptr;
	cout << symbol;
	CHECK(cudaGetSymbolAddress(&symbol, result));
	cout << " " << symbol << endl;
	CHECK(cudaMemcpy(&score, symbol, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << score << endl;
	//char* s = (char*)cudaHead;
	//s += (sizeof(CUDANode*) * 20);
	//int nodeScore;
	//CHECK(cudaMemcpy(&nodeScore, (void*)s, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << "bytes " << score << endl;

	return score;
}