//
// Created by spacegoing on 6/25/16.
//

#ifndef MRFLATENTSSVM_CHECKBOARD_HELPER_H
#define MRFLATENTSSVM_CHECKBOARD_HELPER_H

#include <vector>
#include <math.h>

typedef struct learn_options {
    int K = 4;  //number of lower linear functions
    int gridStep = 16; // grid size for defining cliques
    int maxIters = 100;  //maximum learning iterations
    double eps = 1.0e-16; // constraint violation threshold
    int learningQP = 1;  // encoding for learning QP (1, 2, or 3)
    int figWnd = 0;  // figure for showing results
} LEARN_OPTIONS;

class CheckBoard {
public:

    const int W = 128; // image width
    const int H = 128; // image height
    const LEARN_OPTIONS options;


    const int numCliques = (int) pow((W / options.gridStep), 2);
    const int numVariables = W * H;

    std::vector<std::vector<int>> cliques;
    std::vector<std::vector<int>> y;
    std::vector<std::vector<std::vector<int>>> unary;
    std::vector<std::vector<std::vector<int>>> pairwise;

    CheckBoard();

    void checkboard_generator();


};

#endif //MRFLATENTSSVM_CHECKBOARD_HELPER_H
