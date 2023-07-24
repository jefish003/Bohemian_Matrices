# Bohemian_Matrices
A class for computing the eigenvalues of Bohemian matrices

The Bohemian class currently allows for a number of options of types of matrices to find the eigenvalues of. Currently only {0,1} and {-1,0,1} matrices are available (with the exception of graph Laplacian matrices discussed more below!), however it is on the todo list to expand this. The available options are 'Adjacency', 'Laplacian' and 'Random_Matrix'. Each type is discussed further below.

Getting started - first download the Bohemian.py file included. Be sure the following python packages are available to you: (numpy, matplotlib,time, IPython, tqdm, scipy,datetime,os,platform). NOTE: Bohemian.py has only been currently tested on python 3.8 so if you are having issues you may need to move to a python 3.8 implementation.
Once Bohemian.py is downloaded, make sure you are in the same directory and you can import the class as follows

```
from Bohemian import Bohemian
Boh = Bohemian()
```

Now you are ready to begin. Throughout the rest of the readme file I will assume you have already completed this step.
We will begin by discussing the two classes of graphs ('Adjacency' and 'Laplcian') as each of them has the same available options, the only difference is the type of matrix which results.
If you only want a single instance of a random graph you may use the gen_random_graphs function.

```
A = Boh.gen_random_graphs()
L = Boh.return_Laplacian_matrix()
```
Where A is the Adjacency matrix and L is the graph Laplacian. Note that the defaults without any options passed are the number of nodes 10 (n = 10), simple = True, with simple meaning no self loops, so the adjacency matrix is Hollow (that is has only zeros on the diagonal) and allow_negative_entries = False, meaning that this is a {0,1} adjacency matrix. For the graph Laplacian, the only relevant pieces of information here is n =10 and allow_negative_entries = False, because the graph Laplcian is the same whether or not simple = True. 

Now we have a couple of options for changing the defaults. One way is using the set methods, for instance:

```
Boh.set_n(20)
A = Boh.gen_random_graphs()
L = Boh.return_Laplacian_matrix()
```
will now generate a simple random graph of size 20 nodes (i.e. 20 x 20 adjacency and Laplacian matrix).
Another way to do this same thing is to set the value with n = 20 in the function call,

```
A = Boh.gen_random_graphs(n = 20)
L = Boh.return_Laplacian_matrix()
```

So now we can play around with generating the matrices we want. For instance we may only want the graph Laplacian when entries of {-1,0,1} are allowed in the adjacency matrix (NOTE: The graph Laplacian is computed from the adjacency matrix, which is why A is always computed, and L is only returned if it is asked for)

'''
Boh.gen_random_graphs(n = 20, simple = False, allow_negative_ones = True)
L = Boh.return_Laplacian_matrix()
```
Now it looks like we forgot to return the adjacency matrix! But fortunately the last call is always stored internally so we can retrieve it by using
```
A = Boh.return_adjacency_matrix()
```
Note that for some reason allow_negative_ones is the call in the function, but for setting we actually use:
```
Boh.set_allow_negative_entries(True)
```
poor design choice on my end, but still workable.
