import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import inv
from random import random
from numbers import Number
from scipy.stats import mvn,multivariate_normal
from sklearn.cluster import KMeans
from numpy.linalg import norm
from itertools import permutations
import cvxopt
from collections import OrderedDict
import sys 
from time import time
from abc import ABC, abstractmethod
class PHDabstract(ABC):
    
    """ Abstract class: cannot be used unless its subclasses override all abstract methods """

    def set_params(self,**kwargs):
        """ method to use to set/override a list of attributes """
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
    def reset(self, phd_only = False):
        """ method to reset the phd data - called when new data is generated
            called in prediction with phd_only = True to override predictions only """
        
        if not phd_only:
            #true data
            self.true_data = {}
            self.true_data['targets'] = []
            self.true_data['all_x'] = []
            self.true_data['all_y'] = []
            self.true_data['birth_time'] = {}
            self.true_data['disappear_time'] = {}
            self.true_data['survivors'] = []
            self.true_data['n_targets'] = {}
            self.timeline = []
            self.observed_data = {}
            self.data_generated = 0
            self.data_given = 0

            
        #phd data
        self.phd_filter = {}
        self.phd_filter['weights_br'] = {}
        self.phd_filter['positions_br'] = {}
        self.phd_filter['estimated_positions'] = {}
        self.phd_filter['particles_positions'] = {}
        self.phd_filter['n_targets_predicted'] = {}
        
        self.all_OSPA = {}
        self.all_OMAT = {}

    def set_generated_data(self, phd):
        
        """  Copy the generated data dictionaries from another phd object """
        
        if not phd.data_generated:
            raise ValueError('Data not found in {}'.format(phd))
            
        self.reset()
        
        self.true_data = phd.true_data
        self.timeline = phd.timeline
        self.observed_data = phd.observed_data
        self.n_time_steps = phd.n_time_steps
        self.e = phd.e
        self.region = phd.region
        self.poisson_window = phd.poisson_window
        self.poisson_coef = phd.poisson_coef
        self.poisson_mean = phd.poisson_mean
        self.poisson_cov = phd.poisson_cov

        self.r = phd.r
        self.data_generated = 1
        
    def set_y_obs(self, y_obs):
        
        """ set observed positions """
        
        self.reset()
        self.data_given = 1
        self.observed_data = y_obs
        self.n_time_steps = len(y_obs)
        
    def birth(self):
        # cdf of multivariate gaussian over window [wmin;wmax]
        mu,i = mvn.mvnun(self.poisson_window[0],self.poisson_window[1],self.poisson_mean, self.poisson_cov)
        mu *= self.poisson_coef
        self.mu = mu
        
        # number of newborn targets
        N = np.random.poisson(mu)

        # generate position of N new targets within the window
        positions = []
        for k in range(0,N):
            x_k = np.random.multivariate_normal(mean=self.poisson_mean, cov=self.poisson_cov)
            while (min(x_k>self.poisson_window[0]) == False) or (min(self.poisson_window[1]>x_k) == False):
                x_k = np.random.multivariate_normal(mean=self.poisson_mean, cov=self.poisson_cov)
            positions.append(x_k)

        # return number of newborn targets and positions
        return N, positions 
    
    def add_clutter(self, r):
        
        if not r:
            raise ValueError('r must be > 0')
            
        if not self.data_generated or self.data_given:
            print("No data found to add clutter")
            return None
        
        
        self.r += r

        print('Total clutter is {}'.format(self.r))
        
        for k in range(1,self.n_time_steps+1):
            # Generate false alarms
            n_false_alarms = np.random.poisson(self.r)
            false_alarms = \
            np.array([np.random.uniform(low=self.region[0][0],high=self.region[0][1],size=n_false_alarms),
                      np.random.uniform(low=self.region[1][0],high=self.region[1][1],size=n_false_alarms)])
            
            if k in self.observed_data.keys():
                self.observed_data[k] = np.hstack([self.observed_data[k], false_alarms])
            else:
                self.observed_data[k] = false_alarms
                                               
    def generate(self, override = False):
        
        """ Generates true samples X and their observations Y. As a security measure, 
        if they have already been generated, override must be set to True to override them """
        
        if self.data_generated and not override:
            print('Data have already been generated. To override them use generate(override = True).')
            return None
        
        if self.data_given and not override: 
            print(""" Data have already been given with self.set_yobs. To override them with new generated data,
                  use generate(override = True). """)
            return None
        
        if override:
            self.reset()
                
        for k in range(1,self.n_time_steps+1):

            # Generate false alarms
            n_false_alarms = np.random.poisson(self.r)
            self.observed_data[k] = \
            np.array([np.random.uniform(low=self.region[0][0],high=self.region[0][1],size=n_false_alarms),
                      np.random.uniform(low=self.region[1][0],high=self.region[1][1],size=n_false_alarms)])
            
            
            # Survival or not
            survival = np.random.uniform(size=len(self.true_data['survivors']))
            deads = [self.true_data['survivors'][i] for i in range(len(self.true_data['survivors'])) \
                     if survival[i] >= self.e]
            for i in deads:
                self.true_data['disappear_time'][i] = k
            self.true_data['survivors'] = [self.true_data['survivors'][i] \
                                           for i in range(len(self.true_data['survivors'])) \
                                           if survival[i] < self.e]

            # Birth of new targets
            N, positions = self.birth()
            if N > 0:

                # Add new targets to list of targets and to list of survivors
                if len(self.true_data['targets']) == 0:
                    self.true_data['targets'].extend([i for i in range(N)])
                    self.true_data['survivors'].extend([i for i in range(N)])
                else:
                    list_new_targets = [i for i in \
                                        range(self.true_data['targets'][-1]+1,self.true_data['targets'][-1]+1+N)]
                    self.true_data['targets'].extend(list_new_targets)
                    self.true_data['survivors'].extend(list_new_targets)

                # Add true and observed coordinates of new targets
                for i in range(N):
                    self.true_data['birth_time'][self.true_data['targets'][-N+i]] = k
                    self.true_data['all_x'].append( np.asmatrix(positions[i]) )
                    self.true_data['all_y'].append( np.asmatrix(self.g(positions[i])) )
                    self.observed_data[k] = np.hstack([self.observed_data[k],self.g(positions[i]).reshape(2,1)])


            # List of targets that will leave the region at time step k
            dead_k = []

            for i in self.true_data['survivors']:

                # we only need to compute true and observed coordinates for "old" targets
                if self.true_data['birth_time'][i] != k:

                    # Transition
                    self.true_data['all_x'][i] = \
                    np.vstack([self.true_data['all_x'][i], self.f(self.true_data['all_x'][i][-1,:])])

                    # Checking if the new point is inside the region or not
                    if self.true_data['all_x'][i][-1,0] < self.region[0][0] \
                    or self.true_data['all_x'][i][-1,0] > self.region[0][1] \
                    or self.true_data['all_x'][i][-1,2] < self.region[1][0] \
                    or self.true_data['all_x'][i][-1,2] > self.region[1][1]:

                        self.true_data['all_x'][i] = self.true_data['all_x'][i][:-1,:]

                        dead_k.append(i)
                        self.true_data['disappear_time'][i] = k

                    else:
                        # Observation
                        self.true_data['all_y'][i] = \
                        np.vstack([self.true_data['all_y'][i], self.g(self.true_data['all_x'][i][-1,:])])

                        self.observed_data[k] = \
                        np.hstack([self.observed_data[k],self.g(self.true_data['all_x'][i][-1,:]).reshape(2,1)])

            for j in dead_k:
                self.true_data['survivors'].remove(j)

            # Delete the key in observed_data if there is no observation
            if self.observed_data[k].shape[1] == 0:
                del self.observed_data[k]

        # self.timeline is a list where self.timeline[i] is the list of time steps
        # at which target i is alive
        for i in self.true_data['targets']:
            if i in self.true_data['disappear_time'].keys():
                self.timeline.append(np.arange(self.true_data['birth_time'][i],self.true_data['disappear_time'][i],1))
            else:
                self.timeline.append(np.arange(self.true_data['birth_time'][i],self.n_time_steps+1,1))

        self.timeline = self.timeline

        for k in range(1,self.n_time_steps+1):
            self.true_data['n_targets'][k] = 0
        for target in self.true_data['targets']:
            if target in self.true_data['disappear_time'].keys():
                for k in range(self.true_data['birth_time'][target],self.true_data['disappear_time'][target]):
                    self.true_data['n_targets'][k] += 1
            else:
                for k in range(self.true_data['birth_time'][target],self.n_time_steps+1):
                    self.true_data['n_targets'][k] += 1

        self.data_generated = 1

    def plot_true(self, with_clutter = False):
        
        """Plot the true and observed positions"""
        
        if self.data_generated:
        
            # Choosing a different color for each target
            n_targets = len(self.true_data['targets'])
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 0.9, n_targets)]

            # Plot of the ground truth X vs Y
            fig = plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            if with_clutter and self.r:
                for k in self.observed_data.keys():
                    plt.plot(self.observed_data[k][0], self.observed_data[k][1], 'kx', markersize=1)
                    
            for i in self.true_data['targets']:
                plt.plot(self.true_data['all_y'][i][:,0],self.true_data['all_y'][i][:,1],\
                         'x',label="observed track %s" %i,color=colors[i])
                plt.plot(self.true_data['all_x'][i][:,0],self.true_data['all_x'][i][:,2],\
                         'o-',label="true track %s" %i,color=colors[i])
            plt.xlabel("X",fontsize=20)
            plt.ylabel("Y",fontsize=20)
            #plt.legend(loc='best')

            # self.timeline is a list where self.timeline[i] is the list of time steps
            # at which target i is alive
            #self.timeline = []
            #for i in self.true_data['targets']:
            #    if i in self.true_data['disappear_time'].keys():
            #        self.timeline.append(np.arange(self.true_data['birth_time'][i],self.true_data['disappear_time'][i],1))
            #    else:
            #        self.timeline.append(np.arange(self.true_data['birth_time'][i],41,1))

            # Plot of the ground truth time vs X
            plt.subplot(1,3,2)
            if with_clutter and self.r:
                for k in self.observed_data.keys():     
                    plt.plot(k*np.ones(self.observed_data[k].shape[1]), self.observed_data[k][0], 'kx', markersize=1)
                
            for i in self.true_data['targets']:
                plt.plot(self.timeline[i],self.true_data['all_y'][i][:,0],\
                         'x',label="observed track %s" %i,color=colors[i])
                plt.plot(self.timeline[i],self.true_data['all_x'][i][:,0],\
                         'o-',label="true track %s" %i,color=colors[i])
            plt.xlabel("time",fontsize=20)
            plt.ylabel("X",fontsize=20)
            plt.xlim(0,self.n_time_steps+1)
            #plt.legend(loc='upper right')

            # Plot of the ground truth time vs Y
            plt.subplot(1,3,3)
            if with_clutter and self.r:
                for k in self.observed_data.keys():
                    plt.plot(k*np.ones(self.observed_data[k].shape[1]), self.observed_data[k][1], 'kx', markersize=1)
            for i in self.true_data['targets']:
                plt.plot(self.timeline[i],self.true_data['all_y'][i][:,1],\
                         'x',label="observed track %s" %i,color=colors[i])
                plt.plot(self.timeline[i],self.true_data['all_x'][i][:,2],\
                         'o-',label="true track %s" %i,color=colors[i])
            plt.xlabel("time",fontsize=20)
            plt.ylabel("Y",fontsize=20)
            plt.xlim(0,self.n_time_steps+1)
            #plt.legend(loc='upper right')
            plt.show();

        elif self.data_given:
            raise ValueError("Cannot plot true positions if y_obs is given because the true x are not known.")
        else:
            raise ValueError("No data to plot !")

###########################################################################################################    
    
    def plot_observed(self):
        
        """Plot the observed positions"""
        
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        for k in self.observed_data.keys():
            plt.plot(self.observed_data[k][0], self.observed_data[k][1], 'bx')
        plt.xlabel("X",fontsize=20)
        plt.ylabel("Y",fontsize=20)

        fig = plt.figure(figsize=(16,4))
                                   
        # Plot of time vs X
        plt.subplot(1,3,2)
        for k in self.observed_data.keys():     
            plt.plot(k*np.ones(self.observed_data[k].shape[1]), self.observed_data[k][0], 'bx')
        plt.xlabel("time",fontsize=20)
        plt.ylabel("X",fontsize=20)
        plt.xlim(0,self.n_time_steps+1)

        # Plot of time vs Y
        plt.subplot(1,3,3)
        for k in self.observed_data.keys():
            plt.plot(k*np.ones(self.observed_data[k].shape[1]), self.observed_data[k][1], 'bx')
        plt.xlabel("time",fontsize=20)
        plt.ylabel("Y",fontsize=20)
        plt.xlim(0,self.n_time_steps+1)
        plt.show();

    def plot_true_predictions(self):
        
        """Plot the true and predicted positions"""
        
        if self.data_generated:
            
            # Choosing a different color for each target
            n_targets = len(self.true_data['targets'])
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 0.9, n_targets)]

            # Plot of the ground truth X vs Y
            fig = plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            for k in self.phd_filter['estimated_positions'].keys():
                plt.plot(self.phd_filter['estimated_positions'][k][0], self.phd_filter['estimated_positions'][k][1], 'bx')
            for i in self.true_data['targets']:
                plt.plot(self.true_data['all_x'][i][:,0],self.true_data['all_x'][i][:,2],\
                         '-',label="true track %s" %i,color=colors[i])
            plt.xlabel("X",fontsize=20)
            plt.ylabel("Y",fontsize=20)
            #plt.legend(loc='best')


            # Plot of the ground truth time vs X
            plt.subplot(1,3,2)
            for k in self.phd_filter['estimated_positions'].keys():     
                plt.plot(k*np.ones(self.phd_filter['estimated_positions'][k].shape[1]), self.phd_filter['estimated_positions'][k][0], 'bx')
            for i in self.true_data['targets']:
                plt.plot(self.timeline[i],self.true_data['all_x'][i][:,0],\
                         '-',label="true track %s" %i,color=colors[i])
            plt.xlabel("time",fontsize=20)
            plt.ylabel("X",fontsize=20)
            plt.xlim(0,self.n_time_steps+1)
            #plt.legend(loc='upper right')

            # Plot of the ground truth time vs Y
            plt.subplot(1,3,3)
            for k in self.phd_filter['estimated_positions'].keys():
                plt.plot(k*np.ones(self.phd_filter['estimated_positions'][k].shape[1]), self.phd_filter['estimated_positions'][k][1], 'bx')
            for i in self.true_data['targets']:
                plt.plot(self.timeline[i],self.true_data['all_x'][i][:,2],\
                         '-',label="true track %s" %i,color=colors[i])
            plt.xlabel("time",fontsize=20)
            plt.ylabel("Y",fontsize=20)
            plt.xlim(0,self.n_time_steps+1)
            #plt.legend(loc='upper right')
            plt.show();

        elif self.data_given:
            raise ValueError("Cannot plot true positions if y_obs is given because the true x are not known.")
        else:
            raise ValueError("No data to plot !")
###########################################################################################################    
        
    def plot_observed_predictions(self):
        
        """Plot the observed and predicted positions"""
        
        # Plot of X vs Y
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)  
        for k in self.phd_filter['estimated_positions'].keys():
            plt.plot(self.phd_filter['estimated_positions'][k][0], self.phd_filter['estimated_positions'][k][1], 'bx')
        plt.xlabel("X",fontsize=20)
        plt.ylabel("Y",fontsize=20)
                                   
        # Plot of time vs X
        plt.subplot(1,3,2)
        for k in self.phd_filter['estimated_positions'].keys():     
            plt.plot(k*np.ones(self.phd_filter['estimated_positions'][k].shape[1]), self.phd_filter['estimated_positions'][k][0], 'bx')
        plt.xlabel("time",fontsize=20)
        plt.ylabel("X",fontsize=20)
        plt.xlim(0,self.n_time_steps+1)

        # Plot of time vs Y
        plt.subplot(1,3,3)
        for k in self.phd_filter['estimated_positions'].keys():
            plt.plot(k*np.ones(self.phd_filter['estimated_positions'][k].shape[1]), self.phd_filter['estimated_positions'][k][1], 'bx')
        plt.xlabel("time",fontsize=20)
        plt.ylabel("Y",fontsize=20)
        plt.xlim(0,self.n_time_steps+1)
        plt.show();

    @abstractmethod
    def predict(self):
        pass
    
    def update_step(self, x_new, w, y):

        L=len(x_new[0]) # supposed to be new L, i.e. l_{k} = l_{k-1}+J_k
        n=len(y[0])
        C=np.zeros((1,n))
        weights=np.zeros((1,L))

        # Update of C_k
        for i in range(0,n):
            for l in range(0,L):
                # due to form of observation model
                C[0][i] += w[0][l] * self.g_pdf(y[:,i],x_new[:,l])


        # Update of weights
        for l in range(0,L):
            for i in range(0,n):

                weights[0][l] += self.g_pdf(y[:,i],x_new[:,l]) \
                / (C[0][i] + self.r/((self.region[0][1]-self.region[0][0])*(self.region[1][1]-self.region[1][0])))
			
            weights[0][l] *= w[0][l]

        return weights

    def resample(self, weights, N):

        # returns N indexes from resampling with list weights
        indices = []
        C = np.cumsum(np.array(weights))
        rands = np.random.uniform(size=N)

        for i in range(N):
            j = 0
            while rands[i] > C[j]:
                j += 1
            indices.append(j)

        return indices

    def targets_estimation(self, x, w, N):

        # We predict positions if and only if the estimated number of target N > 0
        if N > 0:

            # number of weights
            L = w.shape[1]

            # control on dimensions coherence
            if x.shape[1] != L:
                raise ValueError("Particles and weights dimensions do not match")
            elif L < N:
                raise ValueError("Not enough intput data to perform clustering")

            # select particles with weights superior or equal to threshold
            if self.threshold == 'mean':
                m = np.mean(w)
            elif self.threshold == 'percentile':
                m = np.percentile(w, self.perc)
            elif isinstance(self.threshold, Number):
                m = self.threshold
            else:
                raise ValueError("'threshold' should be either 'mean', 'percentile' or a number")


            # select indices related to weights higher than m
            indices = [k for k in range(0,L) if w[0,k] >= m]
                
            # Compute clustering of selected particles
            if len(indices) >= N: # remaining particles are enough input for kmeans
                X = np.zeros((2,len(indices)))
                w = w[0,indices]
                X[0,:] = x[0,indices]
                X[1,:] = x[2,indices]

            else: # remaining particles are not enough input for kmeans, return to original array
                X = np.zeros((2,L))
                X[0,:] = x[0,:]
                X[1,:] = x[2,:]
                
            kmeans = KMeans(n_clusters=N,n_jobs=self.n_jobs).fit(X.T)
            labels = kmeans.labels_

            # standard centers, all particles have the same weight
            if self.prediction_method == 'centers':
                result = kmeans.cluster_centers_ 

            # cluster centers weighted with their respective w, in each cluster the weighted average is computed
            elif self.prediction_method == 'max':
                result = np.zeros((N,2))
                for cluster in range(N):
                    cluster_indices = np.where(labels==cluster)[0]
                    w_i = w[cluster_indices]
                    indice_max = w_i.argmax()
                    result[cluster] = X[:,cluster_indices][:,indice_max]
                        
            else:
                raise ValueError(""" prediction_method must be 'centers' or 'max'""")
                
                
        

            # return coordinates of centers, i.e. estimated targets' positions
            return result.T

    def PHDfilter(self, x_k, w_k, y_k1,k):

        # This function computes one iteration of the PHD filter. It shall update the phd_filter
        # dictionnary.
        """Given 
        - x_k : predicted positions at time step k
        - w_k : weights at time step k
        - y_k1 : observed positions at time step k+1
        Returns :
        - x_k1 : predicted positions at time step k+1
        - w_k1 : weights at time step k+1
        """

        # prediction step
        x_k1, w = self.prediction_step(w_k, self.J, x_k, y_k1)

        # update step
        w_k1 = self.update_step(x_k1, w, y_k1)

        # resampling step
        mass = np.sum(w_k1)
        n_part = int(self.rho * mass)
        self.phd_filter['positions_br'][k] = x_k1
        self.phd_filter['weights_br'][k] = w_k1
        indices = self.resample(w_k1/mass, n_part) if n_part > 0 else []
        x_k1 = x_k1[:,indices]
        w_k1 = w_k1[:,indices]

        estimated_x_k1 = self.targets_estimation(x_k1, w_k1, int(round(mass)))

        return x_k1, w_k1, estimated_x_k1, int(round(mass))
    
############# METRICS 

    def OSPA(self, x, y, pnorm, c):
        def dist(x1,y1, pnorm, c):
            return min(c,norm(x1-y1,ord=pnorm))**pnorm

        (d1,m) = x.shape
        (d2,n) = y.shape

        if (m == 0 or n == 0):
            return None
        else:
            if m > n:
                return self.OSPA(y,x,pnorm,c)

            perm = list(permutations(np.arange(n)))
            fact_n = len(perm)
            array = np.zeros(fact_n)
            for i in range(fact_n):
                array[i] = sum(dist(x[:,j],y[:,perm[i][j]], pnorm, c) for j in range(m) )
            index = np.argmin(array)
            return (1/n * (array[index] + c**pnorm *(n-m)))**(1/pnorm)

    @staticmethod
    def OMAT(x, x_pred, pnorm, c):

        (d1,m) = x.shape
        (d2,n) = x_pred.shape

        size = n*m

        if size > 0:

            # distance vector
            distance = np.zeros(size)
            for i in range(n):
                for j in range(m):
                    distance[i*m+j] = norm(x_pred[:,i]-x[:,j])**pnorm

            # matrix for minimization
            P = np.zeros(shape=(n*m,n*m))

            # matrix for inequality constraint
            G = -np.eye(size)

            # vector for inequality constraint
            h = np.zeros(size)

            # matrix for equality constraint
            A = np.zeros(shape=(n+m,size))
            for i in range(n):
                A[i,i*m:(i+1)*m] = 1
            for j in range(m):
                A[n+j,np.arange(0,n*m,m)+j] = 1

            # vector for equality constraint
            b = np.zeros(n+m)
            b[:n] = 1/n
            b[n:m+n] = 1/m

            #return P,distance,G,h,A,b

            # cvxopt
            #P = cvxopt.matrix(P)
            q = cvxopt.matrix(distance)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
            A = cvxopt.matrix(A[0:n+m-1])
            b = cvxopt.matrix(b[0:n+m-1])

            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.lp( q, G, h, A, b)

            return solution['primal objective']**(1/pnorm)
        
    def compute_dico(self):
        dico = {}
        for k in range(1,self.n_time_steps+1):
            dico[k] = np.ndarray(shape=(2,0))
            for tar in range(len(self.timeline)):
                if k in self.timeline[tar]:
                    dico[k] = \
                    np.hstack([dico[k],
                              self.true_data['all_x'][tar][k-np.min(self.timeline[tar]),[[0,2]]].reshape(2,1)])           
            if dico[k].shape[1] == 0:
                del dico[k]
            else:
                dico[k] = np.asarray(dico[k])
                
        return dico
                
    def compute_OSPA(self, dico, pnorm, c, return_criterion = False):
        
        ospa_criterion = max(max(self.phd_filter['n_targets_predicted'].values()),
                             max(self.true_data['n_targets'].values())) <= 8
        all_OSPA = {}
        if ospa_criterion:
            for k in dico.keys() & self.phd_filter['estimated_positions']:
                all_OSPA[k] = self.OSPA(dico[k],self.phd_filter['estimated_positions'][k], pnorm, c)
        
        self.all_OSPA = all_OSPA
        
        
        if return_criterion:
            return ospa_criterion
        
        ospa = np.zeros(self.n_time_steps+1)
        ospa[sorted(self.all_OSPA.keys())] = [self.all_OSPA[k] for k in sorted(self.all_OSPA.keys())]
        
        return ospa[1:]
    
    def get_estimated_number_targets(self):
        
        keys_pred = sorted(self.phd_filter['n_targets_predicted'].keys())
        values_pred = [self.phd_filter['n_targets_predicted'][k] for k in keys_pred]
        keys_true = sorted(self.true_data['n_targets'].keys())
        values_true = [self.true_data['n_targets'][k] for k in keys_true]
        
        pred, true = np.zeros(self.n_time_steps+1), np.zeros(self.n_time_steps+1)
        pred[keys_pred] = values_pred
        true[keys_true] = values_true
                           
        return true[1:], pred[1:]
                           
    def compute_OMAT(self, dico, pnorm, c):
        
        all_OMAT = {}
        for k in dico.keys() & self.phd_filter['estimated_positions']:
            all_OMAT[k] = self.OMAT(dico[k],self.phd_filter['estimated_positions'][k], pnorm, c)
        
        self.all_OMAT = all_OMAT
        
    def wasserstein(self,  pnorm=2, c=10, plot = True, omat = True):
        
        """Compute OSPA and OMAT distances between true positions and predicted
        Plot these graphs as well as the true vs predicted number of targets graph
        Parameters :
        - pnorm : p-norm for OSPA and OMAT
        - c : threshold for OSPA
        """
        
        #dico
        dico = self.compute_dico()
        
        # OSPA
        ospa_criterion = self.compute_OSPA(dico, pnorm, c, return_criterion = True)
        
        # OMAT
        self.compute_OMAT(dico, pnorm, c)

        
        if plot:

            plt.figure(figsize=(16,3))
            plt.xlim(1,self.n_time_steps)
            plt.ylim(0,max(max(self.phd_filter['n_targets_predicted'].values()),
                           max(self.true_data['n_targets'].values()))+1)
            plt.plot(sorted(self.phd_filter['n_targets_predicted'].keys()),
                     [self.phd_filter['n_targets_predicted'][k] for k in sorted(self.phd_filter['n_targets_predicted'].keys())],
                     'r--', label="estimated target number", linewidth=3)
            plt.plot(sorted(self.true_data['n_targets'].keys()),
                     [self.true_data['n_targets'][k] for k in sorted(self.true_data['n_targets'].keys())],
                     'b-', label="true target number",linewidth=1)
            plt.xlabel("time",fontsize=20)
            plt.ylabel("number of targets",fontsize=14)
            plt.legend(loc='best')

            if ospa_criterion:
                plt.figure(figsize=(16,3))
                plt.xlim(0,self.n_time_steps+1)
                plt.ylim(0,int(max(self.all_OSPA.values()))+1)
                plt.plot(sorted(self.all_OSPA.keys()),[self.all_OSPA[k] for k in sorted(self.all_OSPA.keys())],'bo')
                plt.xlabel("time",fontsize=20)
                plt.ylabel("OSPA distance",fontsize=14)

            plt.figure(figsize=(16,3))
            plt.xlim(0,self.n_time_steps+1)
            plt.ylim(0,int(max(self.all_OMAT.values()))+1)
            plt.plot(sorted(self.all_OMAT.keys()),[self.all_OMAT[k] for k in sorted(self.all_OMAT.keys())],'bo')
            plt.xlabel("time",fontsize=20)
            plt.ylabel("OMAT distance",fontsize=14)
            plt.show();

    def _predict(self):
        
        """ Main loop of PHD filter """
        
        # Initialization
        x_k = np.random.rand(4,0)
        w_k = np.ones((1,0))
        
        self.reset(phd_only=True)
        
        # Prediction of the targets' positions
        for k in range(1, self.n_time_steps+1):

            if k in self.observed_data.keys():

                # perform phd filter

                x_k1, w_k1, estimated_x_k1, n_targ_pred = self.PHDfilter(x_k, w_k, self.observed_data[k], k)

                # save predicted positions and update parameters
                self.phd_filter['n_targets_predicted'][k] = n_targ_pred
                self.phd_filter['particles_positions'][k] = x_k1
                if estimated_x_k1 is not None:
                    self.phd_filter['estimated_positions'][k] = estimated_x_k1
                x_k, w_k = np.copy(x_k1), np.copy(w_k1)

            else:
                self.phd_filter['n_targets_predicted'][k] = 0
        
    @staticmethod
    def rescale(data):
        o = 5*(data-data.min())/(data.max()-data.min()) + 1
        return 30*o.astype(int).reshape(-1)+10

    def save_particles(self, path):

        x = np.concatenate(self.true_data['all_x'])
        y = np.concatenate(self.true_data['all_y'])
        true_, pred_ = self.get_estimated_number_targets()
        plt.ioff()

        for i,k in enumerate(self.phd_filter['positions_br'].keys()):
            name = list('0000')
            f = plt.figure(figsize=(25,8))
            w = self.phd_filter['weights_br'][k]
            res = self.rescale(w)
            br = self.phd_filter['positions_br'][k]
            ar = self.phd_filter['particles_positions'][k]

            plt.subplot(1,2,1)
            plt.scatter(br[0,:], br[2,:], s= res, label='Weighted particles' )
            plt.text(x.min()+5, y.max()-2, 'Number of targets: {} - estimated: {}'.format(true_[k-1], pred_[k-1]), fontsize=18)

            plt.xlim([x.min()-5,x.max()+5])
            plt.ylim([y.min()-5,y.max()+5])
            plt.title('Before resampling')

            plt.subplot(1,2,2)
            plt.xlim([x.min()-5,x.max()+5])
            plt.ylim([y.min()-5,y.max()+5])
            plt.scatter(ar[0,:],ar[2,:], s=10, c="b", marker='o', label='Resampled particles')
            plt.title('After resampling')
            for j in range(1,k+1):
                if j in self.phd_filter['estimated_positions'].keys():
                    pr = self.phd_filter['estimated_positions'][j]
                    plt.scatter(pr[0,:], pr[1,:], s=100, c="r", marker='+', label = 'Predicted positions')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor = [2,1])
            n = len(str(i))
            name[-n:] = str(i)
    
            plt.savefig(path+'/{}.png'.format("".join(name)));
            
