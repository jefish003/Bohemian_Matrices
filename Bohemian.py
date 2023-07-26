# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:33:18 2023

@author: fishja
"""

import numpy as np
from matplotlib import pyplot as plt
import time
from IPython.display import display, clear_output
from tqdm import tqdm
from scipy import stats
import datetime
import os
import platform
import matplotlib as mpl
import gc


class Bohemian:
    
    def __init__(self):
        
        self.n = 10
        self.random_matrix = None
        self.adjacency_matrix = None
        self.Laplacian_matrix = None
        self.sparse_storage = False
        self.allow_negative_entries = False
        self.simple = True
        self.markersize = 1
        self.distribution_type = 'uniform'
        self.paramsdict = {'Poisson':1,'neg_bin':[1,0.1]}
        self.batch_process =  False
        self.num_in_batch = 10000
        self.batch_save_name = 'Temp'
        self.batch_recompile = True
        self.Batch_num = None
        self.Now = None
        self.Delete = False
        self.return_determinant = False
        self.return_matrix_norm = False
        self.norm_type = 2
        self.minimum_entry = -1
        self.maximum_entry = 1

    def reset_to_defaults(self):
        self.n = 10
        self.random_matrix = None
        self.adjacency_matrix = None
        self.Laplacian_matrix = None
        self.sparse_storage = False
        self.allow_negative_entries = False
        self.simple = True
        self.markersize = 1
        self.distribution_type = 'uniform'
        self.paramsdict = {'Poisson':1,'neg_bin':[1,0.1]}
        self.batch_process =  False
        self.num_in_batch = 10000
        self.batch_save_name = 'Temp'
        self.batch_recompile = True
        self.Batch_num = None
        self.Now = None
        self.Delete = False
        self.return_determinant = False
        self.return_matrix_norm = False
        self.norm_type = 2
        self.minimum_entry = 0
        self.maximum_entry = 1        
    
    def set_n(self,n):
        self.n = n
        
    def set_sparse_storage(self,TruthVal):
        self.sparse_storage = TruthVal
        
    def set_allow_negative_entries(self,TruthVal):
        self.allow_negative_entries = TruthVal
    
    def set_simple(self,TruthVal):
        self.simple = TruthVal
    
    def set_markersize(self,markersize):
        self.markersize = markersize
    
    def set_distribution_type(self,dist_type):
        self.distribution_type = dist_type
    
    def set_paramsdict(self,dict_of_distribution_params):
        self.paramsdict = dict_of_distribution_params
    
    def set_batch_process(self,TruthVal):
        self.batch_process = TruthVal
    
    def set_num_in_batch(self,num_in_batch):
        self.num_in_batch = num_in_batch
    
    def set_batch_save_name(self,batch_save_name):
        self.batch_save_name = batch_save_name
        
    def set_batch_recompile(self,TruthVal):
        self.batch_recompile = TruthVal
    
    def set_Now(self,Now):
        self.Now = Now
    
    def set_Delete(self,TruthVal):
        self.Delete = TruthVal
    
    def set_return_determinant(self,TruthVal):
        self.return_determinant = TruthVal
        
    def set_return_matrix_norm(self,TruthVal):
        self.return_matrix_norm = TruthVal
        
    def set_norm_type(self,Type):
        self.norm_type = Type
        if Type == 'inf':
            self.norm_type = np.inf
        elif Type == 'nuclear':
            self.norm_type = 'nuc'
        
        elif Type == 'infinity':
            self.norm_type = np.inf
            
    
    def set_minimum_entry(self,minimum_entry):
        self.minimum_entry = minimum_entry
    
    def set_maximum_entry(self,maximum_entry):
        self.maximum_entry = maximum_entry
        
    def return_n(self):
        return self.n
    
    def return_random_matrix(self):
        return self.random_matrix
    
    def return_adjacency_matrix(self):
        return self.adjacency_matrix
    
    def return_Laplacian_matrix(self):
        if self.Laplacian_matrix is None:
            if self.adjacency_matrix is not None:
                if not self.sparse_storage:
                    self.Laplacian_matrix = np.diag(np.sum(self.adjacency_matrix,1))-self.adjacency_matrix
                    
                    return self.Laplacian_matrix
                else:
                    return np.diag(np.sum(self.adjacency_matrix,1))-self.adjacency_matrix
        else:
            return self.Laplacian_matrix
    
    def return_distribution_type(self):
        return self.distribution_type
    
    def return_paramsdict(self):
        return self.paramsdict
    
    def return_batch_save_name(self):
        return self.batch_save_name
    
    def return_Now(self):
        return self.Now
    
    def return_matplotlib_backend(self):
        return self.matplotlib_backend
    
    def return_return_determinant(self):
        return self.return_determinant
    
    def return_return_matrix_norm(self):
        return self.return_matrix_norm
    
    def return_norm_type(self):
        return self.norm_type
    
    def return_minimum_entry(self):
        return self.minimum_entry
    
    def return_maximum_entry(self):
        return self.maximum_entry
    
    def gen_random_matrices(self,n=None,allow_negative_entries=None,distribution_type=None,params=None,minimum_entry=None,maximum_entry=None):
        """Generate random 0,1 or -1,0,1 matrices assuming different distributions. Available 
        distribution types are uniform, poisson (a truncated version in the case of 0,1 and
                                                 a truncated and shifted version in -1,0,1 case)
        and negative binomial (also shifted and truncated as necessary)
        To change the parameter in the Poisson case, """
        if n is not None:
            self.n = n
        
        if allow_negative_entries is not None:
            self.allow_negative_entries = allow_negative_entries
        
        if distribution_type is not None:
            self.distribution_type = distribution_type
            
        if minimum_entry is not None:
            self.minimum_entry = minimum_entry
        
        if maximum_entry is not None:
            self.maximum_entry = maximum_entry 
        
        if params is not None:
            if self.distribution_type == 'Poisson':
                self.paramsdict['Poisson'] = params
            
            elif self.distribution_type == 'neg_bin':
                self.paramsdict['neg_bin'] = params
                
        
        
        if self.allow_negative_entries:
            self.minimum_entry = -1
            self.maximum_entry = 1
            if self.distribution_type == 'uniform':
                A = np.random.randint(self.minimum_entry,self.maximum_entry+1,(self.n,self.n))
                self.random_matrix = A
            
            elif self.distribution_type == 'Poisson':
                Param = self.paramsdict['Poisson']
                Truncval = stats.poisson.cdf(self.maximum_entry+1,Param)
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.poisson.ppf(Unif,Param) +self.minimum_entry
            
            elif self.distribution_type == 'neg_bin':
                Param = self.paramsdict['neg_bin']
                Truncval = stats.nbinom.cdf(self.maximum_entry+1,Param[0],Param[1])
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.nbinom.ppf(Unif,Param[0],Param[1])+self.minimum_entry
                
            else:
                raise ValueError('distribution type must be one of the following uniform, Poisson, neg_bin')
            
        elif self.minimum_entry!=0 or self.maximum_entry!=1:

            if self.distribution_type == 'uniform':
                A = np.random.randint(self.minimum_entry,self.maximum_entry+1,(self.n,self.n))
                self.random_matrix = A
            
            elif self.distribution_type == 'Poisson':
                Param = self.paramsdict['Poisson']
                Truncval = stats.poisson.cdf(self.maximum_entry+1,Param)
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.poisson.ppf(Unif,Param) +self.minimum_entry
            
            elif self.distribution_type == 'neg_bin':
                Param = self.paramsdict['neg_bin']
                Truncval = stats.nbinom.cdf(self.maximum_entry+1,Param[0],Param[1])
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.nbinom.ppf(Unif,Param[0],Param[1])+self.minimum_entry
                
            else:
                raise ValueError('distribution type must be one of the following uniform, Poisson, neg_bin')
                        
            
        else:
            
            if self.distribution_type == 'uniform':
                A = np.random.randint(0,2,(self.n,self.n))
                self.random_matrix = A
            
            elif self.distribution_type == 'Poisson':
                Param = self.paramsdict['Poisson']
                Truncval = stats.poisson.cdf(1,Param)
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.poisson.ppf(Unif,Param) 
                self.random_matrix = A
            
            elif self.distribution_type == 'neg_bin':
                Param = self.paramsdict['neg_bin']
                Truncval = stats.nbinom.cdf(1,Param[0],Param[1])
                Unif = stats.uniform.rvs(0,Truncval,(self.n,self.n))
                A = stats.nbinom.ppf(Unif,Param[0],Param[1])
                
                self.random_matrix = A
                
            else:
                raise ValueError('distribution type must be one of the following uniform, Poisson, neg_bin')
                        
        self.random_matrix = A        
        return A
        
        
                
        
    def gen_random_graphs(self,n=None,simple=None,allow_negative_ones = None,minimum_entry=None,maximum_entry=None):
        """To generate simple random graphs. Simple in the sense
           that no self edges are allowed. The graphs will be stored
           as an adjacency matrix. Note that this method does not 
           account for isomorphisms (that is some graphs may be sampled
                                     with higher probability than others)"""
        if n is not None:
            self.n = n
        
        if allow_negative_ones is not None:
            self.allow_negative_entries = allow_negative_ones
        
        if minimum_entry is not None:
            self.minimum_entry = minimum_entry
        
        if maximum_entry is not None:
            self.maximum_entry = maximum_entry         
        
        
        if simple is not None:
            self.simple=simple
        n = self.n
        A = np.zeros(n**2)
        #Because of python's zero indexing we need n^2+1 below
        #That is we want to allow sampling of the complete graph
        #as well...
        NumEdges = np.random.randint(0,1+n**2)
        PossibleEdges = np.arange(n**2)
        #Shuffle for random selection (to randomize the potential graphs)
        np.random.shuffle(PossibleEdges)
        if self.allow_negative_entries:
            self.minimum_entry=-1
            self.maximum_entry = 1
            if NumEdges>0:
                #A[PosibleEdges[0:NumEdges]] = np.random.randint(self.minimum_entry,self.maximum_entry+1,NumEdges)
                NumNegativeEntries = np.random.randint(0,NumEdges)
                A[PossibleEdges[0:NumNegativeEntries]] = np.random.randint(self.minimum_entry,0,NumNegativeEntries)
                A[PossibleEdges[NumNegativeEntries:NumEdges]] = np.random.randint(1,self.maximum_entry+1,NumEdges-NumNegativeEntries)
        
        elif self.minimum_entry!=0 or self.maximum_entry!=1:
            if NumEdges>0:
                #A[PosibleEdges[0:NumEdges]] = np.random.randint(self.minimum_entry,self.maximum_entry+1,NumEdges)
                NumNegativeEntries = np.random.randint(0,NumEdges)
                A[PossibleEdges[0:NumNegativeEntries]] = np.random.randint(self.minimum_entry,0,NumNegativeEntries)
                A[PossibleEdges[NumNegativeEntries:NumEdges]] = np.random.randint(1,self.maximum_entry+1,NumEdges-NumNegativeEntries)
        
            
        else:
            A[PossibleEdges[0:NumEdges]] = 1
        A = A.reshape((n,n))
        if self.simple:
            A = A - np.diag(np.diag(A))
        
        self.adjacency_matrix = A
        if not self.sparse_storage:
            self.Laplacian_matrix = np.diag(np.sum(A,1))-A
        return A
    
    def live_plot_eigenvalues(self,Type='Adjacency',NumGraphs=100,n=None,simple=None,allow_negative_ones=None,distribution_type=None,params=None, return_determinant=None,return_matrix_norm=None,norm_type=None,minimum_entry=None,maximum_entry=None):
        
        if simple is not None:
            self.simple=simple
        
        if allow_negative_ones is not None:
            self.allow_negative_entries = allow_negative_ones
        
        if distribution_type is not None:
            self.distribution_type = distribution_type
        
        if return_determinant is not None:
            self.return_determinant = return_determinant
        
        if return_matrix_norm is not None:
            self.return_matrix_norm = return_matrix_norm
        
        if norm_type is not None:
            self.set_norm_type(norm_type)
        
        if minimum_entry is not None:
            self.minimum_entry = minimum_entry
        
        if maximum_entry is not None:
            self.maximum_entry = maximum_entry
        
        
        
        Real = np.array([])
        Imag = np.array([])
        if self.return_determinant:
            Det = np.array([])
        
        if self.return_matrix_norm:
            Mat_Norm = np.array([])
        
        if Type=='Adjacency':

            fig,ax = plt.subplots()
            plt.xlabel('Real')
            plt.ylabel('Imaginary')
            if self.simple:
                if self.allow_negative_entries:
                    plt.title('Simple With Negatives Adjacency')
                else:
                    plt.title('Simple Adjacency')
            else:
                if self.allow_negative_entries:
                    plt.title('With Negatives Adjacency')
                else:
                    plt.title('Adjacency')            
            for i in range(NumGraphs):
                self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                Eigs = np.linalg.eigvals(self.adjacency_matrix)
                Real= np.append(Real,np.real(Eigs))
                Imag = np.append(Imag,np.imag(Eigs))
                #Since these are Bohemian matrices, entries are real
                #And thus the imaginary part of the determinant is 0.
                #in exact algebra. Though this is not exact algebra so we will force it to be 
                #real!
                #Also note that the determinant is the product of the eigenvalues, so no need to 
                #recompute
                if self.return_determinant:
                    Det = np.append(Det,np.real(np.prod(Eigs)))
                if self.return_matrix_norm:
                    Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.adjacency_matrix,ord=self.norm_type))                
                ax.cla()
                ax.plot(Real,Imag,'r.',markersize=self.markersize)
                display(fig)
                clear_output(wait=True)

                plt.pause(0.01)

        
        elif Type == 'Laplacian':
            fig,ax = plt.subplots()
            plt.xlabel('Real')
            plt.ylabel('Imaginary')
            if self.simple:
                if self.allow_negative_entries:
                    plt.title('Simple With Negatives Laplacian')
                else:
                    plt.title('Simple Laplacian')
            else:
                if self.allow_negative_entries:
                    plt.title('With Negatives Laplacian')
                else:
                    plt.title('Laplacian')            
            for i in range(NumGraphs):
                self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                Eigs = np.linalg.eigvals(self.Laplacian_matrix)
                Real= np.append(Real,np.real(Eigs))
                Imag = np.append(Imag,np.imag(Eigs))
                if self.return_determinant:
                    Det = np.append(Det,np.real(np.prod(Eigs)))
                    
                if self.return_matrix_norm:
                    Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.Laplacian_matrix,ord=self.norm_type))                      
                ax.cla()
                ax.plot(Real,Imag,'r.',markersize=self.markersize)
                display(fig)
                clear_output(wait=True)

                plt.pause(0.01)
        elif Type == 'Random_Matrix':
            fig,ax = plt.subplots()
            plt.xlabel('Real')
            plt.ylabel('Imaginary')
            if self.allow_negative_entries:
                plt.title(self.distribution_type+ " {-1,0,1} matrices")
            else:
                plt.title(self.distribution_type + " {min_entry,...,max_entry} matrices")
            
            for i in range(NumGraphs):
                self.gen_random_matrices(n=n,allow_negative_entries=allow_negative_ones,distribution_type=distribution_type,params=params,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                Eigs = np.linalg.eigvals(self.random_matrix)
                Real = np.append(Real,np.real(Eigs))
                Imag = np.append(Imag,np.imag(Eigs))
                if self.return_determinant:
                    Det = np.append(Det,np.real(np.prod(Eigs)))
                    
                if self.return_matrix_norm:
                    Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.random_matrix,ord=self.norm_type))                      
                ax.cla()
                ax.plot(Real,Imag,'r.',markersize=self.markersize)
                display(fig)
                clear_output(wait=True)
                plt.pause(0.01)
                
        else:
            raise ValueError("Type must be one of the following Adjacency or Laplacian or Random_Matrix")
        
        if not self.return_determinant and not self.return_matrix_norm:
            return Real,Imag
        elif self.return_determinant and not self.return_matrix_norm:
            #Note that integer matrices have a determinant which is
            #integer valued! So we can round (numerical precision causes these to be
            #very close to integer values but not exactly.)
            return Real,Imag,np.round(Det)
        
        elif self.return_matrix_norm and not self.return_determinant:
            return Real,Imag,Mat_Norm
        else:
            return Real,Imag,np.round(Det),Mat_Norm
    
    def retrieve_eigenvalues(self,Type='Adjacency',NumGraphs=100,n=None,simple=None,allow_negative_ones=None,distribution_type=None,params=None,batch_process=None,num_in_batch=None,batch_recompile=None,batch_save_name=None,return_determinant=None,return_matrix_norm=None,norm_type=None,minimum_entry=None,maximum_entry=None):
        
        Real = np.array([])
        Imag = np.array([])
        
        if return_matrix_norm is not None:
            self.return_matrix_norm =return_matrix_norm
        
        if norm_type is not None:
            self.set_norm_type(norm_type)
           
        if self.return_matrix_norm:
            Mat_Norm = np.array([])
        
        if return_determinant is not None:
            self.return_determinant = return_determinant
        
        if self.return_determinant:
            Det = np.array([])
        
        if batch_process is not None:
            self.batch_process = batch_process
        
        if num_in_batch is not None:
            self.num_in_batch = num_in_batch
        
        if batch_recompile is not None:
            self.batch_recompile = batch_recompile
        
        if batch_save_name is not None:
            self.batch_save_name = batch_save_name
        
        if not self.batch_process:
        
            if Type=='Adjacency':
               
                for i in tqdm(range(NumGraphs)):
                    self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.adjacency_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs))
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))    
                    
                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.adjacency_matrix,ord=self.norm_type))
            
            elif Type == 'Laplacian':
              
                for i in tqdm(range(NumGraphs)):
                    self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.Laplacian_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs))
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))                    
 
                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.Laplacian_matrix,ord=self.norm_type))                    
                    
            elif Type == 'Random_Matrix':
              
                for i in tqdm(range(NumGraphs)):
                    self.gen_random_matrices(n=n,allow_negative_entries=allow_negative_ones,distribution_type=distribution_type,params=params,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.random_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs)) 
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))                    

                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.random_matrix,ord=self.norm_type))                    
    
            else:
                raise ValueError("Type must be one of the following: Adjacency or Laplacian or Random_Matrix")
        
        else:
            Batch_num = 1
            Now = str(datetime.datetime.now()).replace(' ','')
            Now = Now.replace(":","")
            Now = Now.replace(".","")
            self.Now = Now
            if Type=='Adjacency':
               
                for i in tqdm(range(NumGraphs)):
                    Mod = np.mod(i+1,self.num_in_batch)
                    self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.adjacency_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs))
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))                    
                    
                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.adjacency_matrix,ord=self.norm_type))
                        
                    
                    if Mod == 0:
                        if not self.return_determinant and not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag)
                        
                        elif not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det)                            
                        
                        elif not self.return_determinant:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Mat_Norm)
                        
                        else:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det,Mat_Norm)  
                                                      
                        Real = np.array([])
                        Imag = np.array([])
                        if self.return_determinant:
                            Det = np.array([])
                        
                        if self.return_matrix_norm:
                            Mat_Norm = np.array([])
                        Batch_num = Batch_num+1
    
        
            
            elif Type == 'Laplacian':
              
                for i in tqdm(range(NumGraphs)):
                    Mod = np.mod(i+1,self.num_in_batch)
                    self.gen_random_graphs(n=n,simple=simple,allow_negative_ones=allow_negative_ones,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.Laplacian_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs))
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))                    
                     
                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.Laplacian_matrix,ord=self.norm_type))
                                                
                     
                    if Mod == 0:
                        if not self.return_determinant and not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag)
                        
                        elif not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det)                            
                        
                        elif not self.return_determinant:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Mat_Norm)
                        
                        else:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det,Mat_Norm)     
                                         
                        Real = np.array([])
                        Imag = np.array([])
                        if self.return_determinant:
                            Det = np.array([])   
                        
                        if self.return_matrix_norm:
                            Mat_Norm = np.array([])
                        Batch_num = Batch_num+1                    
                    
                    
            elif Type == 'Random_Matrix':
              
                for i in tqdm(range(NumGraphs)):
                    Mod = np.mod(i+1,self.num_in_batch)
                    self.gen_random_matrices(n=n,allow_negative_entries=allow_negative_ones,distribution_type=distribution_type,params=params,minimum_entry=minimum_entry,maximum_entry=maximum_entry)
                    Eigs = np.linalg.eigvals(self.random_matrix)
                    Real= np.append(Real,np.real(Eigs))
                    Imag = np.append(Imag,np.imag(Eigs))  
                    if self.return_determinant:
                        Det = np.append(Det,np.real(np.prod(Eigs)))                    
                        
                    if self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,np.linalg.norm(self.random_matrix,ord=self.norm_type))
                                            
                                  
                    if Mod == 0:
                        if not self.return_determinant and not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag)
                        
                        elif not self.return_matrix_norm:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det)                            
                        
                        elif not self.return_determinant:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Mat_Norm)
                        
                        else:
                            np.savez(self.batch_save_name+'_batch_' +str(Batch_num)+'_'+Now+'.npz',Real,Imag,Det,Mat_Norm)                                            
                        Real = np.array([])
                        Imag = np.array([])
                        if self.return_determinant:
                            Det = np.array([]) 
                        
                        if self.return_matrix_norm:
                            Mat_Norm = np.array([])
                        Batch_num = Batch_num+1                    
                    
            
            else:
                raise ValueError("Type must be one of the following: Adjacency or Laplacian or Random_Matrix")
            
            self.Batch_num = Batch_num
            if self.batch_recompile:
                Real = np.array([])
                Imag = np.array([])
                if self.return_determinant:
                    Det = np.array([])
                
                if self.return_matrix_norm:
                    Mat_Norm = np.array([])
                
                
                for i in range(Batch_num-1):
                    npz = np.load(self.batch_save_name+'_batch_' +str(i+1)+'_'+Now+'.npz')
                    Real = np.append(Real,npz['arr_0'])
                    Imag = np.append(Imag,npz['arr_1'])
                    if self.return_determinant:
                        Det = np.append(Det,npz['arr_2'])
                    
                        if self.return_matrix_norm:
                            Mat_Norm = np.append(Mat_Norm,npz['arr_3'])
                    elif self.return_matrix_norm:
                        Mat_Norm = np.append(Mat_Norm,npz['arr_2'])
                    del npz
                
                for i in range(Batch_num-1):
                    os.remove(self.batch_save_name+'_batch_' +str(i+1)+'_'+Now+'.npz')
                
            else:
                print("Warning: Your batches are saved as files, \n the returned real and imaginary parts are \n only from the final batch, not all values \n to change this behavior, set batch_recompile = True")
        if not self.return_determinant and not self.return_matrix_norm:
            return Real,Imag
        elif self.return_determinant and not self.return_matrix_norm:
            return Real,Imag,np.round(Det)
        elif self.return_matrix_norm and not self.return_determinant:
            return Real,Imag,Mat_Norm
        else:
            return Real,Imag,np.round(Det),Mat_Norm
    
    def plot_large_batch_eigenvalues(self,Now=None,Delete=None,Batch_num=None):
        if Now is not None:
            self.Now = Now
        
        if Delete is not None:
            self.Delete = Delete
        
        if Batch_num is not None:
            self.Batch_num = Batch_num
        
        
        plt.figure(1)
        for i in tqdm(range(self.Batch_num-1)):
            npz = np.load(self.batch_save_name+'_batch_' +str(i+1)+'_'+self.Now+'.npz')
            Real = npz['arr_0']
            Imag = npz['arr_1']
            plt.plot(Real,Imag,'r.',markersize=self.markersize)
            del npz
            del Real
            del Imag
 
            
            if self.Delete:
                os.remove(self.batch_save_name+'_batch_' +str(i+1)+'_'+self.Now+'.npz')
        
        
        
        