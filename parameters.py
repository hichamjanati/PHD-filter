import numpy as np

def pseudo_det(X):
    eig = np.linalg.eig(X)[0]
    return np.prod(eig[eig>0])

T = 1
sd_v1 = 2
sd_v2 = 2
sd_trans = np.array([[sd_v1**2,0],[0,sd_v2**2]])
mat_1 = np.array([[1,T,0,0],[0,1,0,0],[0,0,1,T],[0,0,0,1]])
mat_2 = np.array([[0.5*T**2,0],[T,0],[0,0.5*T**2],[0,T]])
sd_w1 = 2.5
sd_w2 = 2.5
sd_obs = np.array([[sd_w1**2,0],[0,sd_w2**2]])
obs_inv = np.linalg.inv(sd_obs)
mat_3 = np.array([[1,0,0,0],[0,0,1,0]])
sqrt_det_g = np.linalg.det(sd_obs)**(0.5)
cov_f = np.dot(mat_2,np.dot(sd_trans,mat_2.T))
sqrt_pseudo_det_f = pseudo_det(cov_f)**(0.5)
inv_cov_f = np.linalg.pinv(cov_f)
cov_q = np.dot(mat_2,np.dot(sd_trans,mat_2.T))
sqrt_pseudo_det_q = pseudo_det(cov_q)**(0.5)
inv_cov_q = np.linalg.pinv(cov_q)


def f(x):
    x = np.asarray(x).reshape(-1)
    # Returns next true positions and velocities given previous ones (x)
    return np.random.multivariate_normal(mean=np.dot(mat_1,x), cov=cov_f)

def g(x):
    x = np.asarray(x).reshape(-1)
    # Return observed position given true position x
    return np.random.multivariate_normal(mean=np.dot(mat_3,x), cov=sd_obs)

def q(x_k,Y):
    x_k = np.asarray(x_k).reshape(-1)
    mean = np.dot(mat_1,x_k)
    return np.random.multivariate_normal(mean=mean,cov=cov_q)

def q_pdf(y_k,x_k,Y):
    x_k = np.asarray(x_k).reshape(-1)
    mean = np.dot(mat_1,x_k)

    return 1/(2*np.pi*sqrt_pseudo_det_q)*np.exp(-0.5*(y_k-mean).dot(inv_cov_q.dot((y_k-mean).T)))

def f_pdf(x,xk):
    xk = np.asarray(xk).reshape(-1)
    # Returns next true positions and velocities given previous ones (x)
    mean = np.dot(mat_1,xk)
    
    return 1/(2*np.pi*sqrt_pseudo_det_f)*np.exp(-0.5*(x-mean).dot(inv_cov_f.dot((x-mean).T)))

def g_pdf(y,x):
    mean = np.dot(mat_3,x)

    return 1/(2*np.pi*sqrt_det_g)*np.exp(-0.5*(y-mean).dot(obs_inv.dot((y-mean).T)))
