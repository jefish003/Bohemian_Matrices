# Bohemian_Matrices
A class for computing the eigenvalues of Bohemian matrices

The Bohemian class currently allows for a number of options of types of matrices to find the eigenvalues of. Currently only {0,1} and {-1,0,1} matrices are available (with the exception of graph Laplacian matrices discussed more below!), however it is on the todo list to expand this. The available options are 'Adjacency', 'Laplacian' and 'Random_Matrix'. Each type is discussed further below.

Getting started - first download the Bohemian.py file included. Be sure the following python packages are available to you: (numpy, matplotlib,time, IPython, tqdm, scipy,datetime,os,platform). NOTE: Bohemian.py has only been currently tested on python 3.8 so if you are having issues you may need to move to a python 3.8 implementation.
Once Bohemian.py is downloaded, make sure you are in the same directory and you can import the class as follows
```from Bohemian import Bohemian```
