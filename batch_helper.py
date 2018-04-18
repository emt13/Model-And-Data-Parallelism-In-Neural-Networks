# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:42:37 2018

"""

from mpi4py import MPI
import numpy as np

# Moving this inside scatter_data? 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def scatter_data(x,y):
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
    # Shapes...
    shape_x = x.shape
    shape_y = y.shape
    
    N = shape_x[0]
    Dx = int(np.prod(x.shape[1:]))
    Dy = int(np.prod(y.shape[1:]))
    
    # Flatten x and y
    x_flattened = np.reshape(x, (N,Dx))
    y_flattened = np.reshape(y, (N,Dy))
    
    # Some arithmetic...
    n_data_per_rank = N // size
    mod = N % size
    
    """Prepare Send buffers (x and y)"""
    sendbufx = None
    sendbufy = None
    
    # This buffer remembers if there is garbage at the end of each process' sendbuffer[rank] (see comment below)
    sendbuf_garbage = None
    
    if rank == 0:
        sendbufx = np.empty([size, n_data_per_rank+ 1, Dx], dtype='f')
        sendbufy = np.empty([size, n_data_per_rank+ 1, Dy], dtype='f')
        sendbuf_garbage = np.empty([size, 1], dtype='b')
    
        # start and end remember what's the chunk of data to put in sendbuffer
        start = 0
        end = n_data_per_rank+1
        
        # decr is for balance purpose (each process gets the same number of data +/- 1)
        # Processes getting less data have a garbage data at the end. (sendbuf_garbage records this)
        decr = mod
        
        for i in range(size):
            if decr > 0:
                sendbufx[i,:,:] = x_flattened[start:end,:]
                sendbufy[i,:,:] = y_flattened[start:end,:]
                start = start + n_data_per_rank + 1
                end = end + n_data_per_rank + 1   
                sendbuf_garbage[i][0] = False
                   
            else: 
                sendbufx[i,:n_data_per_rank,:] = x_flattened[start:end,:]
                sendbufy[i,:n_data_per_rank,:] = y_flattened[start:end,:]
                start = start + n_data_per_rank
                end = end + n_data_per_rank
                sendbuf_garbage[i][0] = True
                
            decr = decr-1
        # End for
        
    # Back to any process
    """Initialization of receive buffers (x and y)"""
    recvbufx = np.empty([n_data_per_rank+1, Dx], dtype = "f") # dtype = "f" is very important
    recvbufy = np.empty([n_data_per_rank+1, Dy], dtype = "f")
    recvbuf_garbage = np.empty([1], dtype = "b")
    
    """Scatter (x and y)"""
    comm.Scatter(sendbufx, recvbufx, root=0)
    comm.Scatter(sendbufy, recvbufy, root=0)
    comm.Scatter(sendbuf_garbage, recvbuf_garbage, root=0)
    
    if recvbuf_garbage[0]:
        recvbufx = recvbufx[:-1]
        recvbufy = recvbufy[:-1]
    
    # Reshaping
    recvbufx = np.reshape(recvbufx, np.insert(shape_x[1:], 0, recvbufx.shape[0]))
    recvbufy = np.reshape(recvbufy, np.insert(shape_y[1:], 0, recvbufy.shape[0]))

    return(recvbufx, recvbufy)


if __name__=="__main__":
    # Run this with 2 processes
    x = np.random.rand(5, 2, 2)
    y = np.random.rand(5,1)
    a = scatter_data(x,y)
    print("rank: ", rank, "my data: ", a)
    