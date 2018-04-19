# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:42:37 2018

"""

from mpi4py import MPI
import numpy as np

def scatter_data(x, xdims, comm, rank, size):
    """
    For Batch parralelism.
    Scatter data (input and output) from process 0 to all processes.
    The dimensions are conserved (the data are not flat).

    Params:
        x          - Input
                        np.array - Dimensions: N \times [ Dx1, Dx2, ... , Dxm ]
                            
        y          - Output
                        np.array - Dimensions: N \times [ Dy1, Dy2, ... , Dym ]
       
    Returns:
                  - (x_rank, y_rank) Data belonging to process "rank". The scattering is balanced.
                        (np.array , np.array) - Dimensions: (N \times [ Dx1, Dx2, ... , Dxm ] , N \times [ Dy1, Dy2, ... , Dym ])
    """

    buf = None
    recvbuf = None
    
    sizeofdp = xdims[1]
    elementsPerProc = xdims[0]//size
    
    # prep send buffer
    #   this will be done by flattening into a list
    #   then we can easily manipulate into a 2d python array for sending
    if rank == 0:
        buf = []
        tmpX = x.flatten().tolist()
        
        for i in range(size):
            start = elementsPerProc * i * sizeofdp
            end = start + elementsPerProc * sizeofdp
            buf.append(tmpX[start:end])

        # get the leftovers
        ind = 0
        curr = 0
        for elem in tmpX[elementsPerProc*sizeofdp*size:]:
            if curr == sizeofdp:
                ind += 1
                curr = 0
            buf[ind].append(elem)
            curr += 1

    # will recv in a list 
    recvbuf = comm.scatter(buf, root=0)
    
    size_proc = xdims[0] // size
    if rank < xdims[0] % size:
        size_proc += 1

    return np.array(recvbuf).reshape((size_proc, sizeofdp))

def all_reduce_data(dparameter_list, comm, rank, size):
    """
    For Batch parralelism.
    All reduce dw/db from all processes to all processes.
   (The dimensions are conserved (the data are not flat)?)

    Params:
       
    Returns:
    """
    
    sendbuf = dparameter_list
    recvbuf = None
    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    return recvbuf



if __name__=="__main__":
    """
    # Debug scatter_data
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    x = None
    y = None
    if rank == 0:
        #x = np.random.rand(5, 2, 2)
        x = np.random.rand(5,2)
        y = np.random.rand(5,1)
        print(x)

    x = scatter_data(x, (5,2), comm, rank, size)
    y = scatter_data(y, (5,1), comm, rank, size)  
    print("rank: ", rank, "my data: ", x, y)
    """
    
    # Debug All Reduce
    # Run this with 2 processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #Number of layers 
    L = 4
    height_w = 3
    width_w = 2
    
    dw_all_layers = np.random.rand(L, height_w, width_w)
    
    if rank == 0 :
        print("rank: ", rank , "my dw all layers: " ,  dw_all_layers)

    elif rank == 1 :
        print("rank: ", rank , "my dw all layers: " ,  dw_all_layers)

    reduced_dw = all_reduce_data(dw_all_layers, comm, rank, size)

    if rank == 0 :
        print("rank: ", rank, "my reduced dw: ", reduced_dw)
    elif rank == 1 : 
        print("rank: ", rank, "my reduced dw", reduced_dw)