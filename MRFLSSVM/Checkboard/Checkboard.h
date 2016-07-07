//
// Created by spacegoing on 6/29/16.
//

#ifndef CHECKBOARD_CHECKBOARD_H
#define CHECKBOARD_CHECKBOARD_H

#include <armadillo>
#include <math.h>
#include <vector>
#include <random>

using namespace arma;

struct Options {
    // Image Configs
    int gridStep = 16;  // grid size for defining cliques
    int W = 128; // image width
    int H = 128; // image height
    int numCliques = (int) pow(W / gridStep, 2); // number of cliques
    int N = W * H; // number of variables

    int dimUnary = 2;
    int dimPairwise = 3;

    // Learning Configs
    int K = 4;  // number of lower linear functions
    int maxIters = 100;  // maximum learning iterations
    double eps = 1.0e-16;  // constraint violation threshold

    // Other Configs
    int learningQP = 1;  // encoding for learning QP (1, 2, or 3)
    int figWnd = 0;  // figure for showing results
};

class Checkboard {
public:
    Options options;
    Mat<int> cliques;
    Mat<int> y;
    cube unary;
    cube pairwise;

    Checkboard();

    int **mat_to_std_vec(Mat<int> &A);

    float ** mat_to_float_vec(mat &A);

    double ** mat_to_std_vec(mat &A);

    double ***cube_to_std_vec(cube &A);

    void printVector(int **vec);

    void printVector(double **vec);

    void printCube(double ***cube);


private:
    void checkboardHelper();

    mat randomMatrix(int rows, int cols);


};

#endif //CHECKBOARD_CHECKBOARD_H
