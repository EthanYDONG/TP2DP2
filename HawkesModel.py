import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.sparse import csr_matrix
from hkstools.hksimulation.Simulation_Branch_HP import Simulation_Branch_HP
from hkstools.Initialization_Cluster_Basis import Initialization_Cluster_Basis
from hkstools.Kernel_Integration import Kernel_Integration
from hkstools.hksimulation.Kernel import Kernel
from scipy.special import erf

class HawkesModel:
    def __init__(self, Seqs, ClusterNum, Tmax):
        self.Seqs = Seqs
        self.clusternum = ClusterNum
        self.model = None  
        self.Tmax = Tmax

    def Initialization_Cluster_Basis(self, baseType=None, bandwidth=None, landmark=None):
        N = len(self.Seqs)
        D = np.zeros(N)
        for i in range(N):
            D[i] = np.max(self.Seqs[i]['Mark'])
        D = int(np.max(D)) + 1
        self.K = self.clusternum
        self.D = D

        
        if baseType is None and bandwidth is None and landmark is None:
            sigma = np.zeros(N)
            Tmax = np.zeros(N)

            for i in range(N):
                sigma[i] = ((4 * np.std(self.Seqs[i]['Time'])**5) / (3 * len(self.Seqs[i]['Time'])))**0.2
                Tmax[i] = self.Seqs[i]['Time'][-1] + np.finfo(float).eps
            Tmax = np.mean(Tmax)

            self.kernel = 'gauss'  
            self.w = np.mean(sigma)  
            self.landmark = self.w * np.arange(0, np.ceil(Tmax / self.w))  

        
        
        elif baseType is not None and bandwidth is None and landmark is None:
            self.kernel = baseType
            self.w = 1
            self.landmark = 0
        
        elif baseType is not None and bandwidth is not None and landmark is None:
            self.kernel = baseType
            self.w = bandwidth
            self.landmark = 0
        
        else:
            self.kernel = baseType if baseType is not None else 'gauss'
            self.w = bandwidth if bandwidth is not None else 1
            self.landmark = landmark if landmark is not None else 0

        self.alpha = 1
        self.M = len(self.landmark)
        self.beta_a = np.ones((D, self.M, self.K, D)) / (self.M * D**2)   
        self.beta_na = np.ones((D, self.M, 1, D)) / (self.M * D**2)
        
        self.b_a = np.ones((D, self.K)) / D     
        self.b_na = np.ones((D, 1))/D
        
        label = np.ceil(self.K * np.random.rand(1, N)).astype(int)-1
        self.label = label
        
        
        self.R_a = None
        self.R_na = None
        
    def Kernel(self,dt):
        dt = np.array(dt).flatten()
        
        landmarks = np.array(self.landmark)[np.newaxis, :]

        
        dt_tiled = np.tile(dt[:, np.newaxis], (1, len(self.landmark)))
        distance = dt_tiled - landmarks
        if self.kernel == 'exp':
            g = self.w * np.exp(-self.w * distance)
            g[g > 1] = 0

        elif self.kernel == 'gauss':
            g = np.exp(-(distance**2) / (2 * self.w**2)) / (np.sqrt(2 * np.pi) * self.w)
        else:
            print('Error: please assign a kernel function!')
        return g    
            
    def Kernel_Integration(self,dt):
        dt = dt.flatten()
        distance = np.tile(dt[:, np.newaxis], (1, len(self.landmark))) - np.tile(self.landmark, (len(dt), 1))
        landmark = np.tile(self.landmark, (len(dt), 1))
        if self.kernel == 'exp':
            G = 1 - np.exp(-self.w * (distance - landmark))
            G[G < 0] = 0
        elif self.kernel == 'gauss':
            G = 0.5 * (erf(distance / (np.sqrt(2) * self.w)) + erf(landmark / (np.sqrt(2) * self.w)))
        else:
            print('Error: please assign a kernel function!')
            G = None
        return G

    def Loglike(self,cluster_idx,mu_prop = None,A_prop = None,index_prop = None):     
        if mu_prop is  None:
            muest = self.b_a  
        else:
            muest = mu_prop  
        if A_prop is  None:
            Aest = self.beta_a 
        else:
            Aest =  A_prop
        if index_prop is  None:
            indexest = self.label
        else:
            indexest = index_prop

        label_k_indices = np.where(indexest == cluster_idx)[1]
        
        label_k_seqs = [self.Seqs[idx] for idx in label_k_indices]
        Loglikes = []
        i = 0
        for c in range(len(label_k_seqs)):
            Time = label_k_seqs[c]['Time']
            Event = label_k_seqs[c]['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = label_k_seqs[c]['Start']
            if not self.Tmax:
                Tstop = label_k_seqs[c]['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i] 
                ti = Time[i] 
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                
                lambdai = muest[ui_int][cluster_idx] if  mu_prop is None else muest[ui_int]
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)         
                    auiuj = Aest[uj_int, :, cluster_idx, ui_int] if ( A_prop is None) else Aest[uj_int,:,ui_int]
                    pij = gij * auiuj 
                    
                    
                    lambdai = lambdai + np.sum(pij)
                    
                
                Loglike = Loglike - np.log(lambdai)

            Loglike = Loglike + (Tstop - Tstart) * np.sum(muest[:,cluster_idx]) if mu_prop is None else  Loglike + (Tstop - Tstart) * np.sum(muest)
            GK_reshape = np.repeat(GK[:, :, np.newaxis], Aest.shape[3], axis=2) if (A_prop is None) else np.repeat(GK[:, :, np.newaxis], Aest.shape[2], axis=2)
            
            
            Loglike = (Loglike + (GK_reshape * Aest[Event_int, :, cluster_idx, :]).sum()) if (A_prop is None) else (Loglike + (GK_reshape * Aest[Event_int, :, :]).sum())

            Loglikes.append(-Loglike)
        
        loglike_for_allseqinthiscluster = sum(Loglikes)
        return loglike_for_allseqinthiscluster
    

    def loglike_one_a(self,seq_one,cluster_idx):
            Time = seq_one['Time']
            Event = seq_one['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = seq_one['Start']
            if not self.Tmax:
                Tstop = seq_one['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                lambdai = self.b_a[ui_int][cluster_idx]
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    auiuj = self.beta_a[uj_int, :, cluster_idx, ui_int]
                    pij = gij * auiuj
                    lambdai = lambdai + np.sum(pij)     
                Loglike = Loglike - np.log(lambdai)
            
            Loglike = Loglike + (Tstop - Tstart) * np.sum(self.b_a[:,cluster_idx])
           
            GK_reshape = np.repeat(GK[:, :, np.newaxis], self.beta_a.shape[3], axis=2)
            Loglike = Loglike + (GK_reshape * self.beta_a[Event_int, :, cluster_idx, :]).sum()
            return -Loglike
    def loglike_one_na(self,seq_one,cluster_idx):  
            Time = seq_one['Time']
            Event = seq_one['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = seq_one['Start']
            if not self.Tmax:
                Tstop = seq_one['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                
                lambdai = self.b_na[ui_int][cluster_idx] 
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    auiuj = self.beta_na[uj_int, :, cluster_idx, ui_int]
                    pij = gij * auiuj
                    
                    lambdai = lambdai + np.sum(pij)
                    
                Loglike = Loglike - np.log(lambdai)
            
            Loglike = Loglike + (Tstop - Tstart) * np.sum(self.b_na[:,cluster_idx])
            GK_reshape = np.repeat(GK[:, :, np.newaxis], self.beta_na.shape[3], axis=2)
            
            Loglike = Loglike + (GK_reshape * self.beta_na[Event_int, :, cluster_idx, :]).sum()
            return -Loglike

    def update_model(self):
        pass
    
    def loglike_one_a_fordebug(self,seq_one,mean,prec):
            
            
            Time = seq_one['Time']
            Event = seq_one['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = seq_one['Start']
            if not self.Tmax:
                Tstop = seq_one['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                
                lambdai = mean[ui_int]
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    
                    auiuj = prec[uj_int,:,ui_int]
                    pij = gij * auiuj
                    lambdai = lambdai + np.sum(pij)     
                Loglike = Loglike - np.log(lambdai)
            
            Loglike = Loglike + (Tstop - Tstart) * np.sum(self.b_a[:,cluster_idx])
           
            GK_reshape = np.repeat(GK[:, :, np.newaxis], self.beta_a.shape[3], axis=2)
            Loglike = Loglike + (GK_reshape * self.beta_a[Event_int, :, cluster_idx, :]).sum()
            return -Loglike


