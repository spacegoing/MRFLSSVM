#include "Checkboard.h"


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
    unary.slice(1) = 2 * (randu < mat > (options.H, options.W) - 0.5)
                     + eta1 * (1 - y) - eta2 * y;

    pairwise.fill(0);

}

//int main(int argc, char **argv) {
//    Checkboard checkboard;
//    return 0;
//}

