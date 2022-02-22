import numpy as np
from timeit import default_timer as timer



def grad_x(M,i,j):
    return 0.5*( M[i][(j+1)%Nx] - M[i][(j-1)%Nx] ) 

def grad_x_vector(M):
    return 0.5*(np.roll(M,1, axis=1) - np.roll(M,-1, axis=1))

def lapl_2D(M,i,j):
    sides = M[(i+1)%Ny][j] + M[(i-1)%Ny][j] + M[i][(j+1)%Nx] + M[i][(j-1)%Nx]
    corners = M[(i+1)%Ny][(j+1)%Nx] + M[(i-1)%Ny][(j+1)%Nx] + M[(i+1)%Ny][(j-1)%Nx] + M[(i-1)%Ny][(j-1)%Nx]
    return -3*M[i][j] + 0.5*sides + 0.25*corners


def find_side(M, dx, dy):
    return np.roll(np.roll(M, dx, axis = 1), dy, axis = 0)

def lapl_2D_vector(M):
    sides = np.roll(M,1, axis=0) + np.roll(M,-1, axis=0) + np.roll(M,1, axis=1) + np.roll(M,-1, axis=1)
    corners = find_side(M, 1, 1) + find_side(M, -1, 1) + find_side(M, 1, -1) + find_side(M, -1, -1)
    return -3*M + 0.5*sides + 0.25*corners


Nx = Ny = x = 256

a=np.random.randint(np.ones((x, x)))

timer_start = timer()

gr1 = np.zeros((x, x))

for i in range(x):
    for j in range(x):
        gr1[i][j] = grad_x(a, i, j)

print(f'Gradiente Old Way: \t\t{timer()-timer_start:.5f}s')
timer_start = timer()

gr2 = grad_x_vector(a)   

print(f'Gradiente Vettorizzato: \t{timer()-timer_start:.5f}s')

print((gr2 == gr1).all())
