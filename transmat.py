import numpy as np
import matplotlib.pyplot as plt
import sys

def transmat(L=4,v=.1,diff=.1,inject=.2):
    '''
    Creates transition matrix that simulates intraflagellar transport.
    L is the length of the flagellum.
    v is the active transport speed (units per time step).
    time step (s) = v / v(biological (um/s))
    For example, in 0.05s, a motor walking at 2um/s moves at a speed of .1 units/second.
    diff is the diffusion rate (number that stay put = 1-2*diff).
    inject is the fraction that inject.
    Positions 0:L are active transport.
    Positions L+1:2*L are diffusion.
    Position 2*L+1 (end) is the base.
    '''

    mat_len = L*2+1
    M = np.zeros((mat_len,mat_len))

    #build active transport
    for i in range(0,L):
        M[:,i] = np.zeros(mat_len)
        M[i+1,i] = v
        M[i,i] = 1 - v

    #build diffusion
    M[L,L]=.9;
    M[L+1,L]=.1;
    for i in range(L+1,L*2):
        M[i+1,i] = .1;
        M[i,i] = .8;
        M[i-1,i] = .1;


    #build injection
    M[0,-1] = inject
    M[-1,-1] = 1 - inject

    #find principal eigenvector ss
    [w,v] = np.linalg.eig(M)

    ss = v[:,np.argmin(abs(w-1))].real #imaginary part should equal 0, this is just for data format purposes
    ss_scaled = [i*200/sum(ss) for i in ss]

    ava = ss_scaled[-1]*inject/.05 #.05 seconds per time step
    return M, ss_scaled, ava

def equil(L_range=range(1,21), mat_params=(.1,.1,.2), save=False):
    '''
    Function that uses the transmat() function to create matrices with L values
    specified by L_range, then finds the value of the flux going from the base
    into postion 1. This is equal to the steady state number at the base,
    determined by the principal eigenvector, multiplied by the injection
    parameter.

    mat_params: element 0: v, element 1: diff, element 2: inject
    '''

    # equil_ava = np.zeros(len(L_range))
    equil_ava = []
    for i in L_range:
        [m,s,av]=transmat(L=i, v=mat_params[0], diff=mat_params[1], inject=mat_params[2])
        # equil_ava[i]=av
        equil_ava.append(av)

    if save:
        np.savetxt('/Users/student/Box Sync/marshall-lab/transmat_ava.txt',equil_ava)

    return equil_ava


if __name__ == '__main__':
    equil_ava = equil()
