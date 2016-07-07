#include "Checkboard.h"
#include <iostream>
#include <iomanip>

Checkboard::Checkboard() :
        cliques(options.H, options.W),
        y(options.H, options.W),
        unary(options.H, options.W, options.dimUnary),
        pairwise(options.H, options.W, options.dimPairwise) {
    checkboardHelper();
}

void Checkboard::checkboardHelper() {
    bool _black = true;
    int cliqueID = 1;

    //generate ground-truth checkboard
    for (int row = 0; row < options.H; row += options.gridStep) {
        for (int col = 0; col < options.W; col += options.gridStep) {
            cliques(span(row, row + options.gridStep - 1),
                    span(col, col + options.gridStep - 1)).fill(cliqueID);
            cliqueID++;

            y(span(row, row + options.gridStep - 1),
              span(col, col + options.gridStep - 1)).fill(_black ? 0 : 1);
            _black = !_black;
        }
    }

    // generate observed labels
    double eta1 = 0.1;
    double eta2 = 0.1;
    unary.slice(0).fill(0);
    unary.slice(1) = 2 * (randomMatrix(options.H, options.W) - 0.5);
    +eta1 * (1 - y) - eta2 * y;

    pairwise.fill(0);

}

int **Checkboard::mat_to_std_vec(Mat<int> &A) {
    int **V = (int **) malloc(sizeof(int *) * A.n_rows);
    for (size_t i = 0; i < A.n_rows; i++) {
        int *temp = (int *) malloc(sizeof(int) * A.n_cols);
        for (int j = 0; j < A.n_cols; ++j) {
            temp[j] = A(i, j);
        }
        V[i] = temp;
    };
    return V;
}

double ** Checkboard::mat_to_std_vec(mat &A) {
    double **V = (double **) malloc(sizeof(double *) * A.n_rows);
    for (size_t i = 0; i < A.n_rows; i++) {
        double *temp = (double *) malloc(sizeof(double) * A.n_cols);
        for (int j = 0; j < A.n_cols; ++j) {
            temp[j] = A(i, j);
        }
        V[i] = temp;
    };
    return V;
}

float ** Checkboard::mat_to_float_vec(mat &A) {
    float **V = (float **) malloc(sizeof(float *) * A.n_rows);
    for (size_t i = 0; i < A.n_rows; i++) {
        float *temp = (float *) malloc(sizeof(float) * A.n_cols);
        for (int j = 0; j < A.n_cols; ++j) {
            temp[j] = (float) A(i, j);
        }
        V[i] = temp;
    };
    return V;
}

double ***Checkboard::cube_to_std_vec(cube &A) {
    double ***VVV = (double ***) malloc(sizeof(double **) * A.n_rows);
    for (size_t i = 0; i < A.n_rows; i++) {
        double **VV = (double **) malloc(sizeof(double *) * A.n_cols);

        for (size_t j = 0; j < A.n_cols; j++) {

            double *V = (double *) malloc(sizeof(double) * A.n_rows);
            for (int k = 0; k < A.n_slices; ++k) {
                V[k] = A(i, j, k);
            }

            VV[j] = V;
        }

        VVV[i] = VV;
    }

    return VVV;
}

mat Checkboard::randomMatrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);
    mat A(rows, cols);
    for (auto &a : A) {
        a = dist(e2);
    }
    return A;
}

void Checkboard::printVector(int **vec) {
    for (int i = 0; i < options.H; i++) {
        for (int j = 0; j < options.W; j++) {
            std::cout << vec[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void Checkboard::printVector(double **vec) {
    for (int i = 0; i < options.H; i++) {
        for (int j = 0; j < options.W; j++) {
            std::cout << vec[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void Checkboard::printCube(double ***cube) {
    for (int i = 0; i < options.H; i++) {
        for (int j = 0; j < options.W; j++) {
            for (int k = 0; k < options.dimUnary; ++k) {
                std::cout << std::setprecision(2) <<
                cube[i][j][k] << ";";
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }
}


//int main(int argc, char **argv) {
//    Checkboard checkboard;
//    checkboard.printStdVector(
//            checkboard.mat_to_std_vec(checkboard.cliques));
//    checkboard.printStdCube(
//            checkboard.cube_to_std_vec(checkboard.unary));
//
//    return 0;
//}


