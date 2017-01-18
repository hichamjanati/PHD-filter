from base import *


class PHD(PHDabstract):
    
    """ PHD Filter """
    
    def __init__(self,
                 f, f_pdf, g, g_pdf,
                 n_time_steps=40,
                 e=0.95,
                 region=[(-100,100),(-100,100)], 
                 poisson_window=[np.array([-100,-np.float('inf'),-100,-np.float('inf')]),
                                 np.array([100,np.float('inf'),100,np.float('inf')])],
                 poisson_coef=0.2, poisson_mean=np.array([0,3,0,-3]), 
                 poisson_cov=np.array([[10,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,1]]),
                 r=0):
        
        self.reset()
        self.f = f
        self.f_pdf = f_pdf
        self.g = g
        self.g_pdf = g_pdf
        self.n_time_steps = n_time_steps
        self.e = e
        self.region = region
        self.poisson_window = poisson_window
        self.poisson_coef = poisson_coef
        self.poisson_mean = poisson_mean
        self.poisson_cov = poisson_cov
        self.r = r
        
        self.reset_history()
        
    def reset_history(self):
        self.history = {}


    def prediction_step(self, weights, N, x, Y):

        L = x.shape[1] # number of existing targets at previous step
        w = np.zeros((1,L+N)) # new weights
        x_new = np.zeros((4,L+N))

        for i in range(0,L):
            # sample x_new from f, due to choice of q
            x_new[:,i] = self.q(x[:,i],Y)
            # compute new weight, simplification due to the absence of spawning and choice of q
            w[0][i] = self.e * self.f_pdf(x_new[:,i],x[:,i]) / self.q_pdf(x_new[:,i],x[:,i],Y) * weights[0][i] 

        for i in range(L,L+N):
            # sample x_new from normal distribution, due to choice of p
            x_new[:,i] = np.random.multivariate_normal(mean=self.poisson_mean, cov=self.poisson_cov)
            # compute new weights, simplification due to choice of sampling density p
            w[0][i] = self.poisson_coef / N

        return x_new, w
        
    def predict(self, q, q_pdf, rho=1000, J=500, threshold='percentile', perc=10, n_jobs=1, prediction_method='centers'):
        """Returns the predicted positions, the predicted number of targets and their predicted trajectories
        Parameters :
        
        - q : importance sampling random variable generator; takes 2 arguments
        - q_pdf : importance sampling density; takes 3 arguments
        - rho : number of particles generated for each target
        - J : number of particles generated at the beginning of each iteration for the possible birth of new targets
        - threshold : threshold for the selection of the particles with highest weights before kmeans clustering; can be :
            * 'percentile' : select the weights above the perc-th percentile weight
            * 'mean' : select the weights above the mean weight
            * a number
        - perc : percentile if threshold = 'percentile'; should be a float in [0,100]; ignored by other methods
        - n_jobs : nomber of jobs to use for the computation
        """
        
        self.rho = rho
        self.J = J
        self.q = q
        self.q_pdf = q_pdf
        self.threshold = threshold
        self.perc = perc
        self.n_jobs = n_jobs
        self.prediction_method = prediction_method
        
        self._predict()
        
    def performance(self, q, q_pdf, rho=200, J=50, threshold='percentile', perc=10, n_jobs=1,  
                    prediction_method = 'centers',
                 n_size = 1, pnorm = 2, c = 10, return_mean = False,
              keep_phd_data = False, keep_n_targets = False):
        
        self.reset_history()
        self.history['phd'] = []
        
        ospa = np.zeros((n_size,self.n_time_steps))
        true_number = np.zeros((n_size, self.n_time_steps))
        pred_number = np.zeros((n_size, self.n_time_steps))
        
        for i in range(n_size):
            self.predict(q, q_pdf, rho=rho, J=J, threshold='percentile', perc=perc, n_jobs=n_jobs,
                         prediction_method = prediction_method)
            dico = self.compute_dico()
            ospa[i] = self.compute_OSPA(dico, pnorm, c)
            if keep_phd_data:
                self.history['phd'].append(self.phd_filter)
            if keep_n_targets:
                true_number[i], pred_number[i] = self.get_estimated_number_targets()
            
        if return_mean:
            ospa = ospa.mean(axis=0)
        
        self.history['ospa'] = ospa
        self.history['true_targets_number'] = true_number
        self.history['pred_targets_number'] = pred_number

        if keep_n_targets:
            return ospa, true_number, pred_number
        
        return ospa
            
class PHD_bootstrap(PHDabstract):
    
    """ PHD Bootstrap Filter """
    
    def __init__(self,
                 f, f_pdf, g, g_pdf,
                 n_time_steps=40,
                 e=0.95,
                 region=[(-100,100),(-100,100)], 
                 poisson_window=[np.array([-100,-np.float('inf'),-100,-np.float('inf')]),
                                 np.array([100,np.float('inf'),100,np.float('inf')])], 
                 poisson_coef=0.2, poisson_mean=np.array([0,3,0,-3]), 
                 poisson_cov=np.array([[30,0,0,0],[0,1,0,0],[0,0,30,0],[0,0,0,1]]),
                 r=0):
        
        self.reset()
        self.f = f
        self.f_pdf = f_pdf
        self.g = g
        self.g_pdf = g_pdf
        self.n_time_steps = n_time_steps
        self.e = e
        self.region = region
        self.poisson_window = poisson_window
        self.poisson_coef = poisson_coef
        self.poisson_mean = poisson_mean
        self.poisson_cov = poisson_cov
        self.r = r

        self.reset_history()
        
    def reset_history(self):
        self.history = {}


    def prediction_step(self, weights, N, x, Y):
        L = x.shape[1] # number of existing targets at previous step
        w = np.zeros((1,L+N)) # new weights
        x_new = np.zeros((4,L+N))

        for i in range(0,L):
            # sample x_new from f, due to choice of q
            x_new[:,i] = self.f(x[:,i])
            # compute new weight, simplification due to the absence of spawning and choice of q
            w[0][i] = self.e * weights[0][i] 

        for i in range(L,L+N):
            # sample x_new from normal distribution, due to choice of p
            x_new[:,i] = np.random.multivariate_normal(mean=self.poisson_mean, cov=self.poisson_cov)
            # compute new weights, simplification due to choice of sampling density p
            w[0][i] = self.poisson_coef / N

        return x_new, w

    def predict(self, rho=200, J=50,  threshold='percentile', perc=10, n_jobs=1, prediction_method = 'centers'):
        """Returns the predicted positions, the predicted number of targets and their predicted trajectories
        Parameters :
        - rho : number of particles generated for each target
        - J : number of particles generated at the beginning of each iteration for the possible birth of new targets
        - threshold : threshold for the selection of the particles with highest weights before kmeans clustering; can be :
            * 'percentile' : select the weights above the perc-th percentile weight
            * 'mean' : select the weights above the mean weight
            * a number
        - perc : percentile if threshold = 'percentile'; should be a float in [0,100]; ignored by other methods
        - n_jobs : nomber of jobs to use for the computation
        """
        self.rho = rho
        self.J = J
        self.threshold = threshold
        self.perc = perc
        self.n_jobs = n_jobs
        self.prediction_method = prediction_method

        self._predict()
        
    def performance(self, rho=200, J=50, threshold='percentile', perc=10, n_jobs=1, prediction_method = 'centers',
                 n_size = 1, pnorm = 2, c = 10, return_mean = False,
                  keep_phd_data = False, keep_n_targets = False):
        
        self.reset_history()
        self.history['phd'] = []
        
        ospa = np.zeros((n_size,self.n_time_steps))
        true_number = np.zeros((n_size, self.n_time_steps))
        pred_number = np.zeros((n_size, self.n_time_steps))
        
        for i in range(n_size):
            self.predict(rho=rho, J=J, threshold='percentile', perc=perc, n_jobs=n_jobs,
                         prediction_method = prediction_method)
            dico = self.compute_dico()
            ospa[i] = self.compute_OSPA(dico, pnorm, c)
            if keep_phd_data:
                self.history['phd'].append(self.phd_filter)
            if keep_n_targets:
                true_number[i], pred_number[i] = self.get_estimated_number_targets()
            
        if return_mean:
            ospa = ospa.mean(axis=0)
        
        self.history['ospa'] = ospa
        self.history['true_targets_number'] = true_number
        self.history['pred_targets_number'] = pred_number

        if keep_n_targets:
            return ospa, true_number, pred_number
        
        return ospa
            