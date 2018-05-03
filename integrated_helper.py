import numpy as np
from mpi4py import MPI
import sys

def scatter_broadcast(data, datadims, batch_split, nodes_model, nodes_batch, comm, rank, size):
	'''
        Takes in data in the form of a numpy array
        
        We xxx
    '''    
	buf = None
	recvbuf = None

	sizeofdp = datadims[1]
	elementsPerBatchSplit = datadims[0] // nodes_batch

	if rank == 0:
		buf = []
		tmpX = data.flatten().tolist()

		for i in range(size):
			batch_split_i = i // nodes_model
			start = elementsPerBatchSplit * batch_split_i * sizeofdp
			end = start + elementsPerBatchSplit * sizeofdp
			buf.append(tmpX[start:end])

		
		ind = 0
		curr = 0
		for elem in tmpX[elementsPerBatchSplit * sizeofdp * nodes_batch:]:
			if curr == sizeofdp:
				ind += 1
				curr = 0
			for j in range(nodes_model):
				buf[ind*nodes_model+j].append(elem)
			curr += 1

	recvbuf = comm.scatter(buf, root=0)

	size_batch_split = datadims[0] // nodes_batch
	if batch_split < datadims[0] % nodes_batch:
		size_batch_split += 1 


	if not recvbuf:
		return np.array([])
	else: 
		return np.array(recvbuf).reshape((size_batch_split, sizeofdp))

	"""
    rem = datadims[0] % nodes_batch 

    if rank == 0: 
    	sendbuf = []
    	tmpData = data.flatten().tolist()

    	for i in range(size):
    		batch_split_i = i % nodes_batch
    		start = sizeofdp * (elementsPerBatchSplit * batch_split_i + min( rem , batch_split_i ))
    		sizebuf = datadims[0] // nodes_batch
    		if batch_split_i < rem:
    			sizebuf +=1
    		end = start + sizebuf * sizeofdp
    		sendbuf.append(tmpX[start:end])

    recvbuf = comm.scatter(sendbuf, root = 0)
	
    size_proc = 
	"""

if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	nodes_model = 3
	nodes_batch = 2
	batch_split = rank // nodes_model



	x = None
	y = None
	if rank == 0:
        #x = np.random.rand(5, 2, 2)
		x = np.random.rand(5,2)
		y = np.random.rand(5,1)
        print(x)
        comm.Barrier()

	x = scatter_broadcast(x, (5,2), batch_split, nodes_model, nodes_batch, comm, rank, size)
	y = scatter_broadcast(y, (5,1), batch_split, nodes_model, nodes_batch, comm, rank, size)  
	
	if rank == 0:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()

	elif rank == 1:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()

	elif rank == 2:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()

	elif rank == 3:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()

	elif rank == 4:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()
	elif rank == 5:
		print("rank: ", rank, "my data: ", x, y)
		comm.Barrier()