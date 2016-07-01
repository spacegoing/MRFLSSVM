//
// Created by spacegoing on 6/30/16.
//

#ifndef LATENTSSVM_V0_12_MRF_HELPER_H
#define LATENTSSVM_V0_12_MRF_HELPER_H


#include "Checkboard/Checkboard.h"

class MRF{
public:

};


class LSSVM{
    Checkboard checkboard;
    int unaryWeight = 1;
    int pairwiseWeight = 0;
    int featureLength = 2*checkboard.options.K - 1;
    mat linEnvCoeffs = zeros<mat>(featureLength, featureLength);
}
#endif //LATENTSSVM_V0_12_MRF_HELPER_H
