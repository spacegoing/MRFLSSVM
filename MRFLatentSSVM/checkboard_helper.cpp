//
// Created by spacegoing on 6/25/16.
//
#include "checkboard_helper.h"
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <map>
#include <set>
#include <bitset>
#include <iostream>

using namespace std;


CheckBoard::CheckBoard():
        cliques(H,vector<int>(W)),
        y(H,vector<int>(W)),
        unary(H,vector(W,vector<int>(2))),
        pairwise(H,vector(W,vector<int>(3)))
{

}

void CheckBoard::checkboard_generator() {

    bool _black = true;  // indicate _black True or white False
    int _cliqueID = 1;

    for (int row = 0; row < H; row += options.gridStep) {
        for (int col = 0; col < W; col += options.gridStep) {

        }
    }
}




