# -*- coding: utf-8 -*-
import numpy as np
import matlab.engine
import MRF_Helpers as mrf
from Checkboard import Instance, Options
from Utils.ReadMat import loadTestInf, loadMatPairwise
import matplotlib.pyplot as plt

__author__ = 'spacegoing'

eng = matlab.engine.start_matlab()
__DEBUG__ = 'hat'
__plot__ = 1


def quadprog_matlab(P, q, A, b):
    P, q, A, b = [matlab.double(i.tolist())
                  for i in [P, q, A, b]]
    null = matlab.double([])
    theta = eng.quadprog(P, q, A, b, null, null, null, null)

    return np.array(theta, dtype=np.double, order='C').reshape(len(theta))


def cutting_plane_ssvm(theta, vt, instance, options):
    # theta: first 2K - 1 are higher order params, then unary, pairwise and slack
    # vt = Inferred Cutting Plane = truePhi
    # instance = Instance()
    # options = Options()

    history = list()

    ## construct QP objective
    P = np.eye(options.sizePhi + 1, options.sizePhi + 1)
    P[-1, -1] = 0.0  # for slack variable
    q = np.zeros(options.sizePhi + 1)
    q[-1] = 1.0e3  # for slack variable

    ################## Adding Constraints ##########################
    # positivity constraint on pairwise weight
    A = np.zeros([2 * options.K, options.sizePhi + 1], dtype=np.double, order='C')
    b = np.zeros(2 * options.K, dtype=np.double, order='C')
    # options.K-1s' positive constraints for a_k - a_{k+1} -------------------------------------
    # A[0] = [0 -1 0 0...]
    # A[1] = [0 0 -1 0...]
    # A[K-2] = [0 0 0 -1...]
    for i in range(0, options.K - 1):
        A[i, i + 1] = -1
    # options.K-1s' positive constraints for b_{k+1} - b_k -------------------------------------
    # A[K-1] = [0 0 0 0 ... 1 0 0 0 ...]
    # A[K]   = [0 0 0 0 ... 0 1 0 0 ...]
    # A[2*K-3] = [0 0 0 0 ... 0 0 0 1 ...]
    for i in range(options.K - 1, options.sizeHighPhi - 1):
        A[i, i + 1] = 1
    # 1 positive constraints for pairwise
    A[options.sizeHighPhi - 1, options.sizePhi - 1] = 1
    # 1 positive constraints for slack
    A[options.sizeHighPhi, options.sizePhi] = 1
    if __DEBUG__ == 'constraint':
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                print(A[i][j], end=' ')
            print('')

    # Loss augmented inference
    lossUnary = np.zeros([options.H, options.W, 2])
    lossUnary[instance.y == 1, 0] = 1.0 / options.numVariables
    lossUnary[instance.y == 0, 1] = 1.0 / options.numVariables

    violation_old = 0
    ################## iterate until convergence ####################
    for t in range(0, options.maxIters):

        theta = quadprog_matlab(P, q, -A, -b)

        # Decode parameters
        unaryWeight = theta[options.sizeHighPhi]
        pairwiseWeight = max(0, theta[options.sizeHighPhi + 1])
        if options.hasPairwise:
            instance.pairwise[:, 2] = pairwiseWeight

        if __DEBUG__ == 'hat':
            y_hat, z_hat, e_hat = \
                mrf.inf_label_latent_helper(unaryWeight * instance.unary_observed, instance.pairwise,
                                            instance.clique_indexes, theta, options)
            history.append({'theta': theta, 'y_hat': y_hat, 'z_hat': z_hat, 'e_hat': e_hat})
            # for i in range(options.H):
            #     for j in range(options.W):
            #         print(y_hat[i][j], end='')
            #     print('\n')
            # for i in range(options.numCliques):
            #     for j in range(options.K-1):
            #         print(z_hat[i][j], end='')
            #     print('\n')
            if np.sum(z_hat) > 0 and np.sum(z_hat) < 1:
                print(z_hat)
                break

        # infer most violated constraint

        if __DEBUG__ == 'inference':
            # load matlab .mat data
            unary_observed, pairwise, theta, y_inferred, z_inferred, e = loadTestInf()
            y_loss, z_loss, e_loss = \
                mrf.inf_label_latent_helper(unary_observed, pairwise,
                                            instance.clique_indexes, theta, options)
            for i in range(128):
                for j in range(128):
                    if y_loss[i][j] != y_inferred[i][j]:
                        print(str(i) + ' ' + str(j))
            for i in range(64):
                for j in range(9):
                    if z_loss[i][j] != z_inferred.T[i][j]:
                        print(str(i) + str(j))

        y_loss, z_loss, e_loss = \
            mrf.inf_label_latent_helper(unaryWeight * instance.unary_observed - lossUnary, instance.pairwise,
                                        instance.clique_indexes, theta, options)

        # add constraint
        phi = mrf.phi_helper(instance.unary_observed, instance.pairwise, y_loss,
                             z_loss, instance.clique_indexes, options)
        loss = np.sum(y_loss != instance.y) / options.numVariables
        slack = loss - np.dot((phi - vt), theta[:-1])
        violation = slack - theta[-1]

        if violation < options.eps:
            break

        A = np.r_[A, [np.r_[phi - vt, 1]]]
        b = np.r_[b, loss]

    return theta, history


def cccp_outer_loop():
    instance = Instance()
    options = Options()

    if __DEBUG__ == 'matlab':
        unary_observed, pairwise = loadMatPairwise()
        instance.unary_observed = unary_observed
        instance.pairwise = pairwise

    # theta = np.zeros(options.sizePhi + 1, dtype=np.double, order='C')
    # theta[options.sizeHighPhi] = 1  # set unary weight to 1
    theta = np.asarray([np.random.uniform(-1, 1, 1)[0]] + list(-1 * np.random.rand(1, options.K - 1)[0, :]) + \
                       list(np.random.rand(1, options.K - 1)[0, :]) + \
                       [np.random.uniform(-1, 1, 1)[0]] + list(np.random.rand(1, 2)[0, :]),
                       dtype=np.double, order='C')

    for t in range(10):
        theta_old = theta

        latent_inferred = mrf.inf_latent_helper(instance.y, instance.clique_indexes, theta, options)

        if __DEBUG__ == 'matlab':
            black = True
            for i in range(64):
                if i % 8 == 0:
                    black = not black
                latent_inferred[i, :] = 1 if black else 0
                black = not black

        vt = mrf.phi_helper(instance.unary_observed, instance.pairwise,
                            instance.y, latent_inferred, instance.clique_indexes, options)

        theta, history = cutting_plane_ssvm(theta, vt, instance, options)

        if __plot__:
            latent_plot = mrf.inf_latent_helper(instance.y, instance.clique_indexes, theta, options)
            for i in range(len(history)):
                x_y_samples = gen_plot_samples(history[i]['theta'], options)
                plt.plot(x_y_samples[:, 0], x_y_samples[:, 1], '-*')
                # print order text
                plt.text(x_y_samples[np.argmax(x_y_samples[:, 1]), 0],
                         np.max(x_y_samples[:, 1]), str(i))

            nonzero_idx = np.where(np.sum(latent_plot, axis=1) != 0)[0]
            nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
            latent_str = str(latent_plot[nonzero_idx, :])
            plt.title('active latent var: ' + latent_str)

            plt.savefig('iteration_%d' % t, format='svg')
            plt.close()

        if all(theta == theta_old):
            print(latent_inferred)
            print('stop converge at iter: %d' % t)
            print('classification error: %d' % np.sum(np.abs(instance.y - history[-1]['y_hat'])))
            break

        if np.sum(np.abs(instance.y - history[-1]['y_hat'])) / options.numVariables < options.eps:
            print(latent_inferred)
            print('converge at iter: %d' % t)
            print('classification error: %d' % np.sum(np.abs(instance.y - history[-1]['y_hat'])))
            break


def gen_plot_samples(theta, options):
    # decode theta
    a_b_array = np.zeros([options.K, 2])
    a_b_array[0, 0] = theta[0]
    for i in range(1, options.K):
        a_b_array[i, 0] = theta[i] + a_b_array[i - 1, 0]
        a_b_array[i, 1] = theta[i + options.K - 1] + a_b_array[i - 1, 1]

    # generate x-y pairs
    step = 1 / options.K
    x_y_samples = np.zeros([options.K + 1, 2])
    x_y_samples[:-1, 0] = np.arange(0, 1, step)
    x_y_samples[-1, 0] = 1
    for i in range(1, options.K + 1):
        x_y_samples[i, 1] = a_b_array[i - 1, 0] * x_y_samples[i, 0] + a_b_array[i - 1, 1]

    return x_y_samples


if __name__ == "__main__":
    cccp_outer_loop()
