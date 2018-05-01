import numpy as np
from mpi4py import MPI
import sys


def all_gather_data(data, comm, rank, size):
    '''
        Takes in data in the form of a numpy array
        
        We gather the feature shapes of the input so that we can determine the 
            correct size to reshape to after we do the allgather

        The next step then flattens the data and sends it to the rest of the 
            ranks through the allgather function.
    
        The function then creates a numpy array, reshapes it to the correct 
            dimensions, and then returns it.
    '''    

    flat = [data.shape[1]] + data.flatten().tolist()
    
    all_data = comm.allgather(flat)
    
    final = []
    
    width = None
 
    for ad in all_data:
        if len(ad) > 1:
            final.extend(ad[1:])
        else:
            break
        if width is None:
            width = ad[0]
    final = np.array(final).reshape((int(len(final)/width), width))
    
    return final 

def _test_all_gather():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()    

    #data = np.array([]).reshape((0,0))
    data = np.array([[7,8]]).reshape(1, 2)
    if rank == 0:
        data = np.array([[1,2],[3,4],[5,6]])

    gathered = all_gather_data(data, comm, rank, size)
    print(gathered)

def main():
    _test_all_gather()    

if __name__ == "__main__":
    main()
