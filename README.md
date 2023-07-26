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
Where A is the Adjacency matrix and L is the graph Laplacian. Note that the defaults without any options passed are the number of nodes 10 (n = 10), simple = True, with simple meaning no self loops, so the adjacency matrix is Hollow (that is has only zeros on the diagonal) and allow_negative_entries = False, meaning that this is a {minimum_entry,...,maximum_entry} adjacency matrix. The default for minimum_entry is 0 and maximum_entry is 1. IF allow_negative_entries = True then the allowed entries will ALWAYS be {-1,0,1}, allow_negative_entries overrides minimum_entry and maximum_entry. Take note of this behavior, and I will reiterate it later.

Note the graph Laplcian is the same whether or not simple = True. Further NOTE: Technically the Laplacian is not a Bohemian matrix, but it is still fun to look at how the eigenvalues are distributed in this integer valued matrix. This is the only allowed non-Bohemian matrix in the class.

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

```
Boh.gen_random_graphs(n = 20, simple = False, allow_negative_ones = True)
L = Boh.return_Laplacian_matrix()
```
Now it looks like we forgot to return the adjacency matrix! But fortunately the last call is always stored internally so we can retrieve it by using
```
A = Boh.return_adjacency_matrix()
```
Note that we can call this as well:

```
Boh.set_allow_negative_ones(True)
```

Now if we want to change the minimum and maximum entries we can! The simplist scenario is the allow_negative_entries case above, which allows {-1,0,1} matrices. NOTE: IF allow_negative_ones IS SET TO TRUE, minimum_entry and maximum_entry ARE ALWAYS OVERRIDDEN AND SET TO -1 AND 1 RESPECTIVELY!!!
But what if we really want matrices which have entries drawn from {0,1,2}?
We can do that by the following:

```
Boh.set_minimum_entry(0)
Boh.set_maximum_entry(2)
Boh.set_simple(True)
Boh.set_n(15)
A = Boh.gen_random_graphs()
```
You should find the problem here. A has entries that are {-1,0,1} even though we adjusted the minimum to be 0 and the maximum to be 2. 
The previous choices are saved inside the class, so if we left it this way, from now on minimum_entry would be -1, maximum_entry would be 1 BECAUSE WE HAVE LEFT allow_negative_ones to be True. Naturally we could fix this by setting allow_negative_ones to False, but we also have the option of restoring all defaults (which automatically changes allow_negative_ones to False) by making the following call:

```
Boh.reset_to_defaults()
```

Now we will be able to get the {0,1,2} weights we want in the graphs:

```
Boh.set_minimum_entry(0)
Boh.set_maximum_entry(2)
Boh.set_simple(True)
Boh.set_n(15)
A = Boh.gen_random_graphs()
```

Similarly we can generate random matrices with integer valued weights. Currently there are a few options for how to generate the values of the entries, 'uniform', 'Poisson' or 'neg_bin', which correspond to the uniform distribution, the truncated and shifted Poisson distribution and the truncated and shifted negative binomial distribution. TODO: Add more distributions.

Let's suppose we want to generate random matrices of {0,1} entries drawn from a uniform distribution and of size (30 x 30) we can accomplish this by
```
Boh.reset_to_defaults()
A = Boh.gen_random_matrices(n = 30,allow_negative_ones=False)
```
This is because the default distribution type is the uniform distribution. We can change this in one of two ways:
```
Boh.set_distribution_type('Poisson')
A = Boh.gen_random_matrices(n = 30)
```
or
```
A = Boh.gen_random_matrices(n = 30, distribution_type='Poisson')
```
Similarly if we want negative binomial with {-3,...,3} entries
```
A = Boh.gen_random_matrices(n = 30, distribution_type = 'neg_bin',minimum_entry=-3,maximum_entry=3)
```
If we for some reason want to switch back to uniform:

```
A = Boh.gen_random_matrices(n=30, distribution_type = 'uniform')
```
Now the preset parameter for the Poisson distribution is with $\lambda = 1$ and for the negative binomial distribution the parameters are $r = 1, p = 0.1$. However there is an option to set these parameters yourself. The recommended way of doing this is in the function call (recommended because it allows the presets for the other distributions to remain the same, otherwise the other dictionary entries may be removed).

```
A = Boh.gen_random_matrices(n = 15, distribution_type = 'Poisson', params = 0.1)
```
for Poisson, however the negative binomial distribution has multiple parameters so we must pass this as a list:

```
A = Boh.gen_random_matrices(n  = 15, distribution_type = 'neg_bin', params = [2,0.01])
```

Now what we really are interested in is the eigenvalues of these matrices. So we have two options for looking at the eigenvalues, option 1 we can live plot the eigenvalues of the matrices as they are produced (but this option only works well at the moment for a relatively small number of matrices, say 5000) or we can retrieve the eigenvalues and then plot them ourselves afterwards.  We will look at the live plot function first.

Suppose we want to live plot the eigenvalues of 100 (10 x 10) graph Laplacians with entries {0,1} in the adjacency matrix, we can accomplish this by:
(NOTE: TO GET INTERACTIVE LIVE PLOTS IN SPYDER YOU HAVE TO CHANGE THE SETTINGS BY GOING AT THE TOP MENU TO TOOLS -> PREFERENCES -> IPYTHON CONSOLE -> GRAPHICS (THIS IS A TAB AFTER CLICKING ON IPYTHON CONSOLE) AND IN THE GRAPHICS BACKEND CHANGE THIS TO AUTOMATIC FROM INLINE)
```
Boh.reset_to_defaults()
Real,Imag = Boh.live_plot_eigenvalues(Type='Laplacian',NumGraphs=100,n=10)
```
This will plot the eigenvalues as they are produced and the returned Real, Imag will contain the real part and imaginary part of all of the eigenvalues of the 1000 Laplacian matrices when completed. A couple of things to note, the live plotting is resource hungry for some reason, so it is not recommended for large jobs, it should only be used to get an idea of what the plots will look like. Also because of this the batch processing option is not available in the live plotting function.

Now suppose we want to live plot 100 (20 x 20) NON-simple Adjacency matrices with {-1,0,1} entries

```
Boh.reset_to_defaults()
Real,Imag = Boh.live_plot_eigenvalues(Type = 'Adjacency', n = 20, simple = False, allow_negative_ones = True)
```
Note the default is NumGraphs=100.

To finish off examples of live plotting, imagine we want to plot 1000 (10 x 10) random matrices with {-1,0,1} entries, from a truncated Poisson distribution with $\lambda = 0.1$ AND we want to change the size of the marker in the plot to be 0.1 so that it doesn't fill as much space, (Note the below step is pretty slow, so you may wish to skip)

```
Boh.set_markersize(0.1)
Real,Imag = Boh.live_plot_eigenvalues(Type='Random_Matrix',n = 10, NumGraphs=1000,allow_negative_ones=True, distribution_type='Poisson',params=0.1)
```
Obviously this is nice to get an idea of where the eigenvalues fall, but it would be useful to have a much larger number of graphs/matrices. To accomplish this the retrieve_eigenvalues function is available. It will not plot for you, but you can get the values and plot them yourself. Example, suppose we want to plot 5,000 Laplacians of size (6 x 6) with {-2,-1,0,1,2} entries in the adjacency matrix, we can do it by using pyplot as shown below:

```
from matplotlib import pyplot as plt
Real,Imag = Boh.retrieve_eigenvalues(Type = 'Laplacian', n = 6, NumGraphs=5000,minimum_entry=-2,maximum_entry=2)
plt.plot(Real,Imag,'r.',markersize=0.1)
```
FROM HERE ON DOWN I WILL ASSUME YOU HAVE ALREADY IMPORTED PYPLOT

Since I am sampling many graphs, I might be interested in more than just the eigenvalues. There are two other things that can (optionally) be returned, the first one is the determinant of each graph and the second one is the norm of the graph (with options for type of norm). 
So imagine I want to sample (5 x 5) matrices with {-3,...,3} entries and I want the determinant of all 5000 graphs returned, I can do that using the following:

```
Boh.reset_to_defaults()
Real,Imag,Det = Boh.retrieve_eigenvalues(Type='Random_Matrix',n =5, NumGraphs=5000,minimum_entry=-3,maximum_entry=3,return_determinant=True)
plt.figure(1)
plt.plot(Real,Imag,'r.',markersize=0.1)
plt.figure(2)
plt.hist(Det,bins=100)
```

Note that Det WILL NOT BE THE SAME SIZE as Real and Imag in this case. This is because Real and Imag contain all of the eigenvalues of each matrix (so in the above case we have 5 eigenvalues for each matrix, so Real is 5x5000 = 25000 in length). If you want Det to be the same size as Real and Imag, and so that it corresponds appropriately to each eigenvalues, you can use the function repeat from numpy,

```
DetNew = np.repeat(Det,5)
```
now DetNew will correspond to each eigenvalue (and be the same size as Real and Imag). 

Similarly we can ask for the norm, the default norm value is the 2 norm, but there are other options as we will explore momentarily

```
Boh.reset_to_defaults()
Real,Imag,Norm = Boh.retrieve_eigenvalues(Type='Random_Matrix',n =5, NumGraphs=5000,minimum_entry=-3,maximum_entry=3,return_matrix_norm=True)
plt.figure(1)
plt.plot(Real,Imag,'r.',markersize=0.1)
plt.figure(2)
plt.hist(Norm,bins=100)
```

Now what if I don't want the 2 norm? Let's say the Frobenius norm-
```
Norm_I_Want = 'fro'
Boh.reset_to_defaults()
Real,Imag,Norm = Boh.retrieve_eigenvalues(Type='Random_Matrix',n =5, NumGraphs=5000,minimum_entry=-3,maximum_entry=3,return_matrix_norm=True,norm_type=Norm_I_Want)
plt.figure(1)
plt.plot(Real,Imag,'r.',markersize=0.1)
plt.figure(2)
plt.hist(Norm,bins=100)
```

The options for the norm that are available are the following: 1,2, 'inf', 'fro', 'nuc', where 'nuc' is the nuclear norm.

Now I might want to plot 100,000 (10 x 10) random matrices with entries drawn from {0,1}, and the truncated negative binomial distribution with parameters $r =5, p =0.01$. (Note this may take a few minutes) and perhaps I want to label the axes

```
Boh.reset_to_defaults()
Real,Imag = Boh.retrieve_eigenvalues(Type='Random_Matrix', NumGraphs = 100000, distribution_type = 'neg_bin', params=[5,0.01], allow_negative_ones=True)
plt.plot(Real,Imag,'r.',markersize=0.1)
plt.xlabel('Real')
plt.ylabel('Imaginary')
```

Unfortunately even this is kind of slow, we would really like to be able to do millions of these matrices, and preferably quickly. The problem arises from the fact that you are storing all of these eigenvalues in memory, and as more and more memory is being assigned for the eigenvalues the computer begins to slow down the computation. The idea to get around this issue is to perform the computation in batches (also TODO will be to eventually do this in parallel which should help squeeze out even more performance). So essentially we will compute a certain number of matrices eigenvalues, then push that all to disk space to free up memory and keep doing this until the job is complete. The default number of matrices eigenvalues to batch is 10,000, but this is of course changeable.
Some batching examples are below.
We will compute the eigenvalues of 1,000,000 (10 x 10) matrices with entries drawn from a uniform distribution over {0,1}
```
Boh.reset_to_defaults()
Boh.set_batch_process(True)
#Note below we are setting to the default, so this line is not strictly necessary but is to illustrate
Boh.set_num_in_batch(10000)
Real,Imag = Boh.retrieve_eigenvalues(Type = 'Random_Matrix', NumGraphs = 1000000, distribution_type='uniform')
plt.plot(Real,Imag,'r.',markersize=0.05)
```
Now the default is to assume that you still have enough space in memory to store all of the eigenvalues which were computed, so basically in the batching stage temporary files are stored in disk space and then are deleted at the end after being recompiled into large numpy arrays, and only the arrays are returned. 

But what if you can't store everything in memory at the end? Well I have a solution for that as well, essentially the idea is to just keep those temporary files, that is make them permanent! You can do this by toggling the batch_recompile option. So we can do the same thing as we did above but with batch_recompile set to False and furthermore, perhaps we want a cute name to start the files of with (rather than the default of 'Temp') and if you want you can prepend this with a directory of where you want it saved as well.

```
Real,Imag = Boh.retrieve_eigenvalues(Type='Random_Matrix',NumGraphs=1000000,batch_process=True,num_in_batch=10000,batch_recompile=False,batch_save_name='MyBest')
```
Note there is a problem here though, the returned Real and Imag are only the ones from the last batch, NOT all of the eigenvalues. So we need a way to plot these eigenvalues which are now stored on our hard drive. Again we give a couple of options for how to do this. 1) You can simply use the files and plot them yourself, or you can use the builtin plot_large_batch_eigenvalues function. Note that each file is an npz file, so if you want to figure out how to load this yourself you are more than welcome to. 

If you have just run your batched job, like above then you can use the following:
```
Boh.plot_large_batch_eigenvalues()
```

However suppose a few weeks ago you ran the job and you want to plot now

```
#You MUST replace the below Now with your Now from the filename. It is a string of numbers which is easily identifiable in the filename
Now = '2023-07-24194950786627'
#You MUST replace NumofFiles with your last file number (which is the largest number after batch_ in the filename)
NumofFiles = 100
Boh.set_batch_save_name('MyBest')
Boh.plot_large_batch_eigenvalues(Now=Now,Delete=False,Batch_num=NumofFiles)
```

Note setting Delete = True (or Boh.set_Delete(True)) will cause all of those files to be deleted, so choose wisely.
And that is it for now. I hope this is helpful!
