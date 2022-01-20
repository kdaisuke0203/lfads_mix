import os
import datetime
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
#np = torch._np
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import batchify_random_sample, update_param_dict

import pdb

def KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv):
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).sum()
    return klc


def logLikelihoodPoisson(k, lam):
    
    return (k * torch.log(lam) - lam - torch.lgamma(k + 1)).sum()


def logLikelihoodGaussian(x, mu, logvar):
    
    from math import log,pi
    return -0.5*(log(2*pi) + logvar + ((x - mu).pow(2)/torch.exp(logvar))).sum()
    

class LFADS_Net(nn.Module):
    
    def __init__(self, inputs_dim, T, dt,
                 model_hyperparams=None,
                 device = 'cpu', save_variables=False,
                 seed=None):
        
        # call the nn.Modules constructor
        super(LFADS_Net, self).__init__()
        
        # Default hyperparameters
        default_hyperparams  = {### DATA PARAMETERS ###
                                'dataset_name'             : 'unknown',
                                'run_name'                 : 'tmp',
                                
                                ### MODEL PARAMETERS ### 
                                'g_dim'                    : 100,
                                'u_dim'                    : 1, 
                                'factors_dim'              : 20,
                                'g0_encoder_dim'           : 100,
                                'c_encoder_dim'            : 100,
                                'controller_dim'           : 100,
                                'g0_prior_kappa'           : 0.1,
                                'f0_prior_kappa'           : 0.1,
                                'u_prior_kappa'            : 0.1,
                                'f_prior_kappa'            : 0.1,
                                'keep_prob'                : 1.0,
                                'clip_val'                 : 5.0,
                                'max_norm'                 : 200,
            
                                ### OPTIMIZER PARAMETERS 
                                'learning_rate'            : 0.01,
                                'learning_rate_min'        : 1e-5,
                                'learning_rate_decay'      : 0.95,
                                'scheduler_on'             : True,
                                'scheduler_patience'       : 6,
                                'scheduler_cooldown'       : 6,
                                'epsilon'                  : 0.1,
                                'betas'                    : (0.9, 0.99),
                                'l2_gen_scale'             : 0.0,
                                'l2_con_scale'             : 0.0,
                                'kl_weight_schedule_start' : 0,
                                'kl_weight_schedule_dur'   : 2000,
                                'l2_weight_schedule_start' : 0,
                                'l2_weight_schedule_dur'   : 2000,
                                'ew_weight_schedule_start' : 0,
                                'ew_weight_schedule_dur'   : 2000}
        
        # Store the hyperparameters        
        self._update_params(default_hyperparams, model_hyperparams)
        
        self.inputs_dim                = inputs_dim
        self.T                         = T
        self.dt                        = dt

        self.device                    = device
        self.save_variables            = save_variables
        self.seed                      = seed
        
        if self.seed is None:
            self.seed = random.randint(1, 10000)
            print('Random seed: {}'.format(self.seed))
        else:
            print('Preset seed: {}'.format(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        
        # Store loss
        self.full_loss_store = {'train_loss' : {}, 'train_recon_loss' : {}, 'train_kl_loss' : {},'train_dir_loss' : {},
                                'valid_loss' : {}, 'valid_recon_loss' : {}, 'valid_kl_loss' : {},
                                'l2_loss' : {}}
        self.train_loss_store = []
        self.valid_loss_store = []
        self.best = np.inf
        
        # Training variable
        self.epochs = 0
        self.current_step = 0
        self.last_decay_epoch = 0
        self.cost_weights = {'kl' : {'weight': 0, 'schedule_start': self.kl_weight_schedule_start,
                                     'schedule_dur': self.kl_weight_schedule_dur},
                             'l2' : {'weight': 0, 'schedule_start': self.l2_weight_schedule_start,
                                     'schedule_dur': self.l2_weight_schedule_dur}}
        
        
        # Generator Forward Encoder
        self.gru_Egen_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
        
        # Generator Backward Encoder
        self.gru_Egen_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
        
        # Controller Forward Encoder
        self.gru_Econ_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
        
        # Controller Backward Encoder
        self.gru_Econ_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
        
        # Controller
        self.gru_controller    = nn.GRUCell(input_size= self.c_encoder_dim * 2 + self.factors_dim, hidden_size= self.controller_dim)
        
        # Generator
        self.gru_generator     = nn.GRUCell(input_size= self.u_dim, hidden_size= self.g_dim)
        
        self.gru_dynamics_mean     = nn.GRUCell(input_size= self.factors_dim, hidden_size= self.factors_dim)
        self.gru_dynamics_var     = nn.GRUCell(input_size= self.factors_dim, hidden_size= self.factors_dim)
        self.gru_dynamics    = nn.GRUCell(input_size= self.factors_dim, hidden_size= self.g_dim)
        
        
        self.fc_g0mean   = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features= self.g_dim)
        self.fc_g0logvar = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features= self.g_dim)
        
        # mean and logvar of the posterior distribution for the inferred inputs (u provided to g)
        # takes as inputs:
        #  - the controller at time t (c_t)
        self.fc_umean   = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
        self.fc_ulogvar = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
        
        # factors from generator output
        self.fc_factors_mean = nn.Linear(in_features= self.g_dim, out_features= self.factors_dim)
        self.fc_factors_var = nn.Linear(in_features= self.g_dim, out_features= self.factors_dim)
        
        self.fc_factors_prior_mean = nn.Linear(in_features= self.g_dim, out_features= self.factors_dim)
        self.fc_factors_prior_var = nn.Linear(in_features= self.g_dim, out_features= self.factors_dim)
        
        # logrates from factors
        #self.fc_logrates   = nn.Linear(in_features= self.factors_dim, out_features= self.inputs_dim)
        self.fc_logrates = nn.Sequential(nn.Linear(self.factors_dim,200),nn.ReLU(),
                                          nn.Dropout(0.2),
                                  nn.Linear(200,200),nn.ReLU(),
                                         nn.Dropout(0.2),
                                  nn.Linear(200,self.inputs_dim))
        # -----------
        # Dropout layer
        # -----------
        self.dropout = nn.Dropout(1.0 - self.keep_prob)
        
        
        # Step through all layers and adjust the weight initiazition method accordingly
        for m in self.modules():
            
            # GRU layer, update using input weight and recurrent weight dimensionality
            if isinstance(m, nn.GRUCell):
                k_ih = m.weight_ih.shape[1] # dimensionality of the inputs to the GRU
                k_hh = m.weight_hh.shape[1] # dimensionality of the GRU outputs
                m.weight_ih.data.normal_(std = k_ih ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
                m.weight_hh.data.normal_(std = k_hh ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
            
            # FC layer, update using input dimensionality
            elif isinstance(m, nn.Linear):
                k = m.in_features # dimensionality of the inputs
                m.weight.data.normal_(std = k ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))

        # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
        self.fc_factors_mean.weight.data = F.normalize(self.fc_factors_mean.weight.data, dim=1)
        
        # --------------------------
        # LEARNABLE PRIOR PARAMETERS INIT
        # --------------------------
        
        self.g0_prior_mu = nn.parameter.Parameter(torch.tensor(0.0))
        self.f0_prior_mu = nn.parameter.Parameter(torch.tensor(0.0))
        self.u_prior_mu  = nn.parameter.Parameter(torch.tensor(0.0))
        self.f_prior_mu  = nn.parameter.Parameter(torch.tensor(0.0))
        #self.hidden_dyna_prior  = nn.parameter.Parameter(torch.tensor(0.0))
        
        from math import log
        self.g0_prior_logkappa = nn.parameter.Parameter(torch.tensor(log(self.g0_prior_kappa)))
        self.f0_prior_logkappa = nn.parameter.Parameter(torch.tensor(log(self.f0_prior_kappa)))
        self.u_prior_logkappa  = nn.parameter.Parameter(torch.tensor(log(self.u_prior_kappa)))
        self.f_prior_logkappa  = nn.parameter.Parameter(torch.tensor(log(self.f_prior_kappa)))
    
        self.optimizer = opt.Adam(self.parameters(), lr=self.learning_rate, eps=self.epsilon, betas=self.betas)
        
        self.logLikelihood = logLikelihoodPoisson
        
        
    def initialize(self, batch_size=None):
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        self.g0_prior_mean = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_mu            # g0 prior mean
        self.f0_prior_mean = torch.ones(batch_size, self.factors_dim).to(self.device)*self.f0_prior_mu            # gf0 prior mean
        self.u_prior_mean  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_mu             # u prior mean
        self.f_prior_mean  = torch.ones(batch_size, self.factors_dim).to(self.device)*self.f_prior_mu             # f prior mean
        #self.hidden_dyna  = torch.ones(batch_size, self.g_dim).to(self.device)*self.hidden_dyna_prior            # f prior mean
        
        self.g0_prior_logvar = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_logkappa    # g0 prior logvar
        self.f0_prior_logvar = torch.ones(batch_size, self.factors_dim).to(self.device)*self.f0_prior_logkappa    # f0 prior logvar
        self.u_prior_logvar  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_logkappa     # u prior logvar
        self.f_prior_logvar  = torch.ones(batch_size, self.factors_dim).to(self.device)*self.f_prior_logkappa     # u prior logvar
        
        self.c = Variable(torch.zeros((batch_size, self.controller_dim)).to(self.device))  # Controller hidden state
        self.hidden_dyna = Variable(torch.zeros((batch_size, self.g_dim)).to(self.device))  # dynamics hidden state
        self.efgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Forward generator encoder
        self.ebgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Backward generator encoder
        
        self.efcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Forward controller encoder
        self.ebcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Backward controller encoder
            
        if self.save_variables:
            self.factors       = torch.zeros(batch_size, self.T, self.factors_dim)
            self.inputs        = torch.zeros(batch_size, self.T, self.u_dim)
            self.inputs_mean   = torch.zeros(batch_size, self.T, self.u_dim)
            self.inputs_logvar = torch.zeros(batch_size, self.T, self.u_dim)
            self.rates         = torch.zeros(batch_size, self.T, self.inputs_dim)

    def encode(self, x):
        
        # Dropout some data
        if self.keep_prob < 1.0:
            x = self.dropout(x)
        
        for t in range(1, self.T+1):
            
            # generator encoders
            self.efgen = torch.clamp(self.gru_Egen_forward(x[:, t-1], self.efgen), max=self.clip_val)
            self.ebgen = torch.clamp(self.gru_Egen_backward(x[:, -t], self.ebgen), max=self.clip_val)
            
            # controller encoders
            self.efcon[:, t]      = torch.clamp(self.gru_Econ_forward(x[:, t-1], self.efcon[:, t-1].clone()),max=self.clip_val)
            self.ebcon[:, -(t+1)] = torch.clamp(self.gru_Econ_backward(x[:, -t], self.ebcon[:, -t].clone()),max=self.clip_val)
        
        # Concatenate efgen_T and ebgen_1 for generator initial condition sampling
        egen = torch.cat((self.efgen, self.ebgen), dim=1)

        # Dropout the generator encoder output
        if self.keep_prob < 1.0:
            egen = self.dropout(egen)
            
        # Sample initial conditions for generator from g0 posterior distribution
        self.g0_mean   = self.fc_g0mean(egen)
        self.g0_logvar = torch.clamp(self.fc_g0logvar(egen), min=np.log(0.0001))
        self.g         = Variable(torch.randn(self.batch_size, self.g_dim).to(self.device))*torch.exp(0.5*self.g0_logvar)\
                         + self.g0_mean
        
        # KL cost for g(0)
#         pdb.set_trace()
        self.kl_loss   = KLCostGaussian(self.g0_mean, self.g0_logvar,
                                        self.g0_prior_mean, self.g0_prior_logvar)/x.shape[0]
        
        # Initialise factors
        self.f         = self.fc_factors_mean(self.g)
        self.f0_mean = self.fc_factors_mean(self.g)
        self.f0_logvar = torch.clamp(self.fc_factors_var(self.g), min=np.log(0.0001))
        self.klf_loss   = KLCostGaussian(self.f0_mean, self.f0_logvar,
                                        self.f0_prior_mean, self.f0_prior_logvar)/x.shape[0]
        
        
    def generate(self, x):
        
        self.recon_loss = 0
        self.dir_loss = 0
        f_his = []
        f_his_mean = []
        f_his_logvar = []
        b_size = x.shape[0]
        for t in range(self.T):
            
            # Concatenate ebcon and efcon outputs at time t with factors at time t+1 as input to controller
            # Note: we take efcon at t+1, because the learnable biases are at first index for efcon
            econ_and_fac = torch.cat((self.efcon[:, t+1].clone(), self.ebcon[:,t].clone(), self.f), dim = 1)

            # Dropout the controller encoder outputs and factors
            if self.keep_prob < 1.0:
                econ_and_fac = self.dropout(econ_and_fac)
            
            # Update controller with controller encoder outputs
            self.c = torch.clamp(self.gru_controller(econ_and_fac, self.c), min=0.0, max=self.clip_val)

            # Calculate posterior distribution parameters for inferred inputs from controller state
            self.u_mean   = self.fc_umean(self.c)
            self.u_logvar = self.fc_ulogvar(self.c)

            # Sample inputs for generator from u(t) posterior distribution
            self.u = Variable(torch.randn(self.batch_size, self.u_dim).to(self.device))*torch.exp(0.5*self.u_logvar) \
                        + self.u_mean

            self.kl_loss = self.kl_loss + KLCostGaussian(self.u_mean, self.u_logvar,
                                        self.u_prior_mean, self.u_prior_logvar)/x.shape[0]

            # Update generator
            self.g = torch.clamp(self.gru_generator(self.u,self.g), min=0.0, max=self.clip_val)

            # Dropout on generator output
            if self.keep_prob < 1.0:
                self.g = self.dropout(self.g)
            
            # Generate factors from generator state
            self.f_mean   = self.fc_factors_mean(self.g)
            self.f_logvar = self.fc_factors_var(self.g)
            self.f = Variable(torch.randn(self.batch_size, self.factors_dim).to(self.device))*torch.exp(0.5*self.f_logvar) \
                        + self.f_mean
            f_his.append(self.f)
            f_his_mean.append(self.f_mean)
            f_his_logvar.append(self.f_logvar)
            
            """if t>0:
                #self.f_prior_mean   = torch.clamp(self.gru_dynamics_mean(f_his_mean[t-1],self.f_prior_mean), min=0.0, max=self.clip_val)
                #self.f_prior_logvar = torch.clamp(self.gru_dynamics_var(f_his_logvar[t-1],self.f_prior_logvar), min=0.0, max=self.clip_val)
                self.hidden_dyna = torch.clamp(self.gru_dynamics(f_his_mean[t-1],self.hidden_dyna), min=0.0, max=self.clip_val)
                self.f_prior_mean1   = self.fc_factors_prior_mean(self.hidden_dyna)
                self.f_prior_logvar1 = self.fc_factors_prior_var(self.hidden_dyna)
                self.klf_loss = self.klf_loss + KLCostGaussian(self.f_mean, self.f_logvar,
                                            self.f_prior_mean1, self.f_prior_logvar1)/x.shape[0]
                for i in range(b_size):
                    self.dir_loss += 10*(abs((f_his[t][i][0]-f_his[t-1][i][0])/(f_his[t-1][i][0]+1e-6))*(x[i,t,0]+x[i,t,1]+x[i,t,2]+x[i,t,3]+x[i,t,4]+x[i,t,5]+x[i,t,9]+x[i,t,10]+x[i,t,12]+x[i,t,16]+x[i,t,21]) +
                                        abs((f_his[t][i][1]-f_his[t-1][i][1])/(f_his[t-1][i][1]+1e-6))*(x[i,t,8]+x[i,t,20]+x[i,t,24]+x[i,t,28])).sum()"""
            #print("x",x[:,t,:],"WW",x[1,t,0])
            #self.dir_loss += 10*(abs(self.f[:,1] - 100)).sum() 
            # Generate rates from factor state
            self.r = torch.exp(self.fc_logrates(self.f))
            
            self.recon_loss = self.recon_loss - self.logLikelihood(x[:, t-1], self.r * self.dt)/x.shape[0]
                  
            if self.save_variables:
                self.factors[:, t] = self.f.detach().cpu()
                #print("WWWWWW",self.factors)
                self.inputs[:, t] = self.u.detach().cpu()
                self.inputs_mean[:, t] = self.u_mean.detach().cpu()
                self.inputs_logvar[:, t] = self.u_logvar.detach().cpu()
                self.rates[:, t]   = self.r.detach().cpu()
                        
    def forward(self, x):
        
        batch_size, steps_dim, inputs_dim = x.shape
        
        assert steps_dim  == self.T
        assert inputs_dim == self.inputs_dim
        
        self.batch_size = batch_size
        self.initialize(batch_size=batch_size)
        self.encode(x)
        self.generate(x)
        
    def infer_trj(self, x):
        pred_data = self.reconstruct(x)
        #self.encode(x)
        infer_trj = []
        for t in range(self.T):
            econ_and_fac = torch.cat((self.efcon[:, t+1].clone(), self.ebcon[:,t].clone(), self.f), dim = 1)
            self.c = torch.clamp(self.gru_controller(econ_and_fac, self.c), min=0.0, max=self.clip_val)

            self.u_mean   = self.fc_umean(self.c)
            self.u_logvar = self.fc_ulogvar(self.c)

            self.u = Variable(torch.randn(self.batch_size, self.u_dim).to(self.device))*torch.exp(0.5*self.u_logvar) \
                        + self.u_mean

            self.g = torch.clamp(self.gru_generator(self.u,self.g), min=0.0, max=self.clip_val)

            self.f_mean   = self.fc_factors_mean(self.g)
            self.f_logvar = self.fc_factors_var(self.g)
            self.f = Variable(torch.randn(self.batch_size, self.factors_dim).to(self.device))*torch.exp(0.5*self.f_logvar) \
                        + self.f_mean
            infer_trj.append(self.f)
        return infer_trj
    
    def reconstruct(self, x):
        
        self.eval()
        self.batch_size = x.shape[0]
        prev_save = self.save_variables
        with torch.no_grad():
            self.save_variables = True
            self(x)
        
        self.save_variables = prev_save # reset to previous value
        
        return self.rates.mean(dim=0).cpu().numpy()
    
    def mcdropout(self, x, state, T=100):
        
        state = Variable(torch.from_numpy(state).type(torch.FloatTensor).cuda())
        self.batch_size = x.shape[0]
        self.train()
        prev_save = self.save_variables
        with torch.no_grad():
            self.save_variables = True
            self(x)
        Yt_hat = np.array([torch.exp(self.fc_logrates(state)).data.cpu().numpy() for _ in range(T)]).squeeze()
        
        self.save_variables = prev_save # reset to previous value
        
        return Yt_hat
    
    def transition_rnn(self, state):
        state = Variable(torch.from_numpy(state).type(torch.FloatTensor).cuda())
        v = torch.randn(state.shape[0],self.g_dim).cuda()
        print(state.shape)
        v = torch.clamp(self.gru_dynamics(state, v), min=0.0, max=self.clip_val)
        mu   = self.fc_factors_prior_mean(v).data.cpu().numpy()
        var = self.fc_factors_prior_var(v).data.cpu().numpy()
        
        return mu, var
    
    
    def infer_factors(self, x):
        
        self.eval()
        self.batch_size = x.shape[0]
        prev_save = self.save_variables
        with torch.no_grad():
            self.save_variables = True
            self(x)
        
        self.save_variables = prev_save # reset to previous value
        return self.factors.detach().cpu()

    
    def weight_schedule_fn(self, step):
        
        for cost_key in self.cost_weights.keys():
            # Get step number of scheduler
            weight_step = max(step - self.cost_weights[cost_key]['schedule_start'], 0)
            
            # Calculate schedule weight
            self.cost_weights[cost_key]['weight'] = min(weight_step/ self.cost_weights[cost_key]['schedule_dur'], 1.0)
    
    
    def apply_decay(self, current_loss):
        
        if len(self.train_loss_store) >= self.scheduler_patience:
            if all((current_loss > past_loss for past_loss in self.train_loss_store[-self.scheduler_patience:])):
                if self.epochs >= self.last_decay_epoch + self.scheduler_cooldown:
                    self.learning_rate  = self.learning_rate * self.learning_rate_decay
                    self.last_decay_epoch = self.epochs
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.learning_rate
                    print('Learning rate decreased to %.8f'%self.learning_rate)
            
    
    def test(self, l2_loss, dl=None, dataset=None, batch_size=4):
        self.eval()
        if dl is None:
            if dataset is None:
                raise IOError('Must pass either a dataset or a dataloader.')
            else:
                self.batch_size = batch_size
                dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        elif dataset is not None:
            print('If both a dataloader and a dataset are passed, the\ndataloader is used.')
            
        else:
            self.batch_size = dl.batch_size

        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
            
        for i, x in enumerate(dl, 0):
            with torch.no_grad():
                x = Variable(x[0])
                self(x)
                loss = self.recon_loss + self.kl_loss + l2_loss + self.dir_loss + self.klf_loss
                test_loss += loss.data
                test_recon_loss += self.recon_loss.data
                test_kl_loss += self.kl_loss.data               
                
        test_loss /= (i+1)
        test_recon_loss /= (i+1)
        test_kl_loss /= (i+1)
        return test_loss, test_recon_loss, test_kl_loss, 
    
    def fit(self, train_dataset, valid_dataset,
            batch_size=4, max_epochs=100,
            use_tensorboard=False, health_check=False,
            train_truth=None, valid_truth=None, output='.'):
        self.batch_size = batch_size
        
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size)
        
        # Initialize directory to save checkpoints
        save_loc = '%s/models_sim/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)

        # Create model_checkpoint directory if it doesn't exist
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        elif os.path.exists(save_loc) and self.epochs==0:
            os.system('rm -rf %s'%save_loc)
            os.makedirs(save_loc)
            
        # Initialize tensorboard
        if use_tensorboard:
            tb_folder = '%s/models_sim/%s/%s/tensorboard/'%(output, self.dataset_name, self.run_name)
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            elif os.path.exists(tb_folder) and self.epochs==0:
                os.system('rm -rf %s'%tb_folder)
                os.mkdir(tb_folder)
            
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(tb_folder)
            
        print('Beginning training...')
        loss_his = []
        for epoch in range(max_epochs):
            self.train()
            if self.learning_rate <= self.learning_rate_min:
                break
            
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            train_klf_loss = 0
            train_dir_loss = 0
            
            # for each batch...
            for i, x in enumerate(train_dl, 0):
                self.current_step += 1
                                                
                # apply Variable wrapper to batch
                x = Variable(x[0])

                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Calculate regularizer weights
                self.weight_schedule_fn(self.current_step)
                
                # Forward
                self(x)
                
                # Calculate l2 regularisation penalty
                l2_loss = self.l2_gen_scale * self.gru_generator.weight_hh.norm(2)/self.gru_generator.weight_hh.numel() + \
                          self.l2_con_scale * self.gru_controller.weight_hh.norm(2)/self.gru_controller.weight_hh.numel() + self.l2_con_scale * self.gru_dynamics.weight_hh.norm(2)/self.gru_dynamics.weight_hh.numel()
                
                # Collect separate weighted losses
                kl_weight = self.cost_weights['kl']['weight']
                l2_weight = self.cost_weights['l2']['weight']
                loss = self.recon_loss + kl_weight * self.kl_loss + l2_weight * l2_loss #+ kl_weight * self.klf_loss#+ self.dir_loss
                
                # Check if loss is nan
                assert not torch.isnan(loss.data), 'Loss is NaN'
                                
                loss.backward()
                
                # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
                
                # update the weights
                self.optimizer.step()

                # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
                self.fc_factors_mean.weight.data = F.normalize(self.fc_factors_mean.weight.data, dim=1)
                
                if use_tensorboard:
                    self.health_check(writer)
                
                train_loss += loss.data
                train_recon_loss += self.recon_loss.data
                train_kl_loss += self.kl_loss.data 
                #train_klf_loss += self.klf_loss.data 
                #train_dir_loss += self.dir_loss.data 
            
            train_loss /= (i+1)
            train_recon_loss /= (i+1)
            train_kl_loss /= (i+1)
            #train_klf_loss /= (i+1)
            #train_dir_loss /= (i+1)
            loss_his.append(train_loss)
            #valid_loss, valid_recon_loss, valid_kl_loss = self.test(l2_loss, dl=valid_dl)
        
            print('Epoch: %4d, Step: %5d, training loss: %.3f' %(self.epochs+1, self.current_step, train_loss))
            print('recon: %4d, kl: %5d, dir: %4d, klf: %4d, klw: %4d' %(self.recon_loss, kl_weight * self.kl_loss, self.dir_loss, kl_weight *self.klf_loss,kl_weight))
            
            # Apply learning rate decay function
            if self.scheduler_on:
                self.apply_decay(train_loss)
                
            # Store loss
            self.train_loss_store.append(float(train_loss))
            #self.valid_loss_store.append(float(valid_loss))
            
            self.full_loss_store['train_loss'][self.epochs]       = float(train_loss)
            self.full_loss_store['train_recon_loss'][self.epochs] = float(train_recon_loss)
            self.full_loss_store['train_kl_loss'][self.epochs]    = float(train_kl_loss)
            self.full_loss_store['train_dir_loss'][self.epochs]    = float(train_dir_loss)
            #self.full_loss_store['valid_loss'][self.epochs]       = float(valid_loss)
            #self.full_loss_store['valid_recon_loss'][self.epochs] = float(valid_recon_loss)
           # self.full_loss_store['valid_kl_loss'][self.epochs]    = float(valid_kl_loss)
            self.full_loss_store['l2_loss'][self.epochs]          = float(l2_loss.data)
            
            # Write results to tensorboard
            if use_tensorboard:
                
                writer.add_scalars('1_Loss/1_Total_Loss', {'Training' : float(train_loss), 
                                                         'Validation' : float(valid_loss)}, self.epochs)

                writer.add_scalars('1_Loss/2_Reconstruction_Loss', {'Training' :  float(train_recon_loss), 
                                                                  'Validation' : float(valid_recon_loss)}, self.epochs)
                
                writer.add_scalars('1_Loss/3_KL_Loss' , {'Training' : float(train_kl_loss), 
                                                       'Validation' : float(valid_kl_loss)}, self.epochs)
                
                writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss.data), self.epochs)
                
                writer.add_scalar('2_Optimizer/1_Learning_Rate', self.learning_rate, self.epochs)
                writer.add_scalar('2_Optimizer/2_KL_weight', kl_weight, self.epochs)
                writer.add_scalar('2_Optimizer/3_L2_weight', l2_weight, self.epochs)

            self.epochs += 1
            
            # Save model checkpoint if training error hits a new low and kl and l2 loss weight schedule
            # has completed
            """if self.current_step >= max(self.cost_weights['kl']['schedule_start'] + self.cost_weights['kl']['schedule_dur'],
                                        self.cost_weights['l2']['schedule_start'] + self.cost_weights['l2']['schedule_dur']):
                if self.valid_loss_store[-1] < self.best:
                    self.last_saved = epoch
                    self.best = self.valid_loss_store[-1]
                    # saving checkpoint
                    self.save_checkpoint(output=output)
                    
                    if use_tensorboard:
                        figs_dict_train = self.plot_summary(data= train_dataset.tensors[0], truth= train_truth)
                        writer.add_figure('Examples/1_Train', figs_dict_train['traces'], self.epochs, close=True)
                        writer.add_figure('Factors/1_Train', figs_dict_train['factors'], self.epochs, close=True)
                        writer.add_figure('Inputs/1_Train', figs_dict_train['inputs'], self.epochs, close=True)

                        figs_dict_valid = self.plot_summary(data= valid_dataset.tensors[0], truth= valid_truth)
                        writer.add_figure('Examples/2_Valid', figs_dict_valid['traces'], self.epochs, close=True)
                        writer.add_figure('Factors/2_Valid', figs_dict_valid['factors'], self.epochs, close=True)
                        writer.add_figure('Inputs/2_Valid', figs_dict_valid['inputs'], self.epochs, close=True)
                        
                        if train_truth is not None:
                            writer.add_figure('Ground_truth/1_Train', figs_dict_train['truth'], self.epochs, close=True)
                            writer.add_figure('R-squared/1_Train', figs_dict_train['rsq'], self.epochs, close=True)
                        
                        if valid_truth is not None:
                            writer.add_figure('Ground_truth/2_Valid', figs_dict_valid['truth'], self.epochs, close=True)
                            writer.add_figure('R-squared/2_Valid', figs_dict_valid['rsq'], self.epochs, close=True)"""
                        
        if use_tensorboard:
            writer.close()
        
        import pandas as pd
        df = pd.DataFrame(self.full_loss_store)
        df.to_csv('%s/models_sim/%s/%s/loss.csv'%(output, self.dataset_name, self.run_name), index_label='epoch')
        
        #self.save_checkpoint(force=True, output=output)
        plt.plot(loss_his)
        print('...training complete.')
    
    
    def plot_traces(self, pred, true,figsize=(8,8), num_traces=12, ncols=2, mode=None, norm=True, pred_logvar=None):
        '''
        Plot fitted intensity function and compare to ground truth
        
        Arguments:
            - pred (np.array): array of predicted values to plot (dims: num_steps x num_cells)
            - true (np.array)   : array of true values to plot (dims: num_steps x num_cells)
            - figsize (2-tuple) : figure size (width, height) in inches (default = (8, 8))
            - num_traces (int)  : number of traces to plot (default = 24)
            - ncols (int)       : number of columns in figure (default = 2)
            - mode (string)     : mode to select subset of traces. Options: 'activity', 'rand', None.
                                  'Activity' plots the the num_traces/2 most active traces and num_traces/2
                                  least active traces defined sorted by mean value in trace
            - norm (bool)       : normalize predicted and actual values (default=True)
            - pred_logvar (np.array) : array of predicted values log-variance (dims: num_steps x num_cells) (default= None)
        
        '''
        
        num_cells = pred.shape[-1]
        
        nrows = int(num_traces/ncols)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        axs = np.ravel(axs)
        
        if mode == 'rand':  
            idxs  = np.random.choice(list(range(num_cells)), size=num_traces, replace=False)
            idxs.sort()
                
        elif mode == 'activity':
            idxs = true.max(axis=0).argsort()[-num_traces:]
        
        else:
            idxs  = list(range(num_cells))
        
        time = np.arange(0, self.T*self.dt, self.dt)
        
        def zscore(x):
            return (x - x.mean())/x.std()
        
        if norm:
            ztrue = zscore(true)
            zpred = zscore(pred)
        else:
            ztrue = true
            zpred = pred
            
        zmin = min(zpred[:, idxs].min(), ztrue[:, idxs].min())
        zmax = max(zpred[:, idxs].max(), ztrue[:, idxs].max())
        
        if np.any(pred_logvar):
            pred_stdev = np.exp(0.5*pred_logvar)
            zmin = min(zmin, (zpred[:, idxs] - pred_stdev[:, idxs]).min())
            zmax = max(zmax, (zpred[:, idxs] + pred_stdev[:, idxs]).max())
        
        for ii, (ax,idx) in enumerate(zip(axs,idxs)):
            plt.sca(ax)
            plt.plot(time, zpred[:, idx], lw=2, color='#37A1D0')
            if np.any(pred_logvar):
                plt.fill_between(time, zpred[:, idx] + pred_stdev[:, idx], zpred[:, idx] - pred_stdev[:, idx], color= '#37A1D0', alpha=0.5, zorder=-2)
            plt.plot(time, ztrue[:, idx], lw=2, color='#E84924')
            plt.ylim(zmin-(zmax-zmin)*0.1, zmax+(zmax-zmin)*0.1)
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            if ii >= num_traces - ncols:
                plt.xlabel('time (s)', fontsize=14)
                plt.xticks(fontsize=12)
                ax.xaxis.set_ticks_position('bottom')
                
            else:
                plt.xticks([])
                ax.xaxis.set_ticks_position('none')
                ax.spines['bottom'].set_visible(False)

            if ii%ncols==0:
                plt.yticks(fontsize=12)
                ax.yaxis.set_ticks_position('left')
            else:
                plt.yticks([])
                ax.yaxis.set_ticks_position('none')
                ax.spines['left'].set_visible(False)
                
        fig.subplots_adjust(wspace=0.1, hspace=0.1)        
        return fig
    
    
    def plot_factors(self, max_in_col=2, figsize=(8,8)):
        
        nrows = max_in_col
        ncols = int(np.ceil(self.factors_dim/max_in_col))
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        axs = np.ravel(axs)
        time = np.arange(0, self.T*self.dt, self.dt)
        factors = self.factors.mean(dim=0).cpu().numpy()
        print("FF",self.factors.shape)
        fmin = factors.min()
        fmax = factors.max()
        
        for jx in range(self.factors_dim):
            plt.sca(axs[jx])
            plt.plot(time, factors[:, jx])
            plt.ylim(fmin-0.1, fmax+0.1)
            
            if jx%ncols == 0:
                plt.ylabel('position')
            else:
                plt.ylabel('')
                axs[jx].set_yticklabels([])
            
            if (jx - jx%ncols)/ncols == (nrows-1):
                plt.xlabel('Time (s)')
            else:
                plt.xlabel('')
                axs[jx].set_xticklabels([])
        
        fig.suptitle('Factors 1-%i for a sampled trial.'%factors.shape[1])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        
        return fig
   
    
    def plot_inputs(self, fig_width=8, fig_height=1.5):
    
        figsize = (fig_width, fig_height*self.u_dim)
        fig, axs = plt.subplots(nrows=self.u_dim, figsize=figsize)
        fig.suptitle('Input to the generator for a sampled trial', y=1.2)
        inputs = self.inputs_mean.mean(dim=0).cpu().numpy()
        time = np.arange(0, self.T*self.dt, self.dt)
        for jx in range(self.u_dim):
            if self.u_dim > 1:
                plt.sca(axs[jx])
            else:
                plt.sca(axs)
            plt.plot(time, inputs[:, jx])
            plt.xlabel('time (s)')
        return fig

    
    def plot_rsquared(self, pred, true, figsize=(6, 4)):
        fig = plt.figure(figsize=figsize)
        var = true.var()
        mse = ((pred - true)**2).mean()
        rsq = 1-mse/var
        
        plt.plot(np.ravel(true), np.ravel(pred), '.')
        plt.xlabel('Ground Truth Rate (Hz)')
        plt.ylabel('Inferred Rates (Hz)')
        
        plt.title('R-squared coefficient = %.3f'%rsq)
        
        return fig
    
    
    def plot_summary(self, data, truth=None, num_average=100):
        
        plt.close()
        figs_dict = {}
        
        batch_example, ix = batchify_random_sample(data, num_average)
        
        pred_data = self.reconstruct(batch_example)
        true_data = data[ix].cpu().numpy()
        figs_dict['traces'] = self.plot_traces(pred_data, true_data, mode='activity', norm=False)
        figs_dict['traces'].suptitle('Spiking Data vs.Inferred Rate')
        figs_dict['traces'].legend(['Inferred Rates', 'Spikes'])
        
        """if torch.is_tensor(truth):
            pred_rate = self.rates.mean(dim=0).cpu().numpy()
            true_rate = truth[ix].cpu().numpy()
            figs_dict['truth'] = self.plot_traces(pred_rate, true_rate, mode='rand')
            figs_dict['truth'].suptitle('Inferred vs. Ground-truth rate functions')
            figs_dict['truth'].legend(['Inferred', 'Ground-truth'])
            figs_dict['rsq']   = self.plot_rsquared(pred_rate, true_rate)"""
            
        figs_dict['factors'] = self.plot_factors()
        #figs_dict['inputs']  = self.plot_inputs()
        
        return figs_dict

    
    def save_checkpoint(self, force=False, purge_limit=50, output='.'):
        '''
        Save checkpoint of network parameters and optimizer state
        
        Arguments:
            force (bool) : force checkpoint to be saved (default = False)
            purge_limit (int) : delete previous checkpoint if there have been fewer
                                epochs than this limit before saving again
        '''
        
        # output_filename of format [timestamp]_epoch_[epoch]_loss_[training].pth:
        #  - timestamp   (YYMMDDhhmm)
        #  - epoch       (int)
        #  - loss        (float with decimal point replaced by -)
        
        save_loc = '%s/models_sim/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)

        if force:
            pass
        else:
            if purge_limit:
                # Get checkpoint filenames
                try:
                    _,_,filenames = list(os.walk(save_loc))[0]
                    split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
                    epochs = [att[2] for att in split_filenames]
                    epochs.sort()
                    last_saved_epoch = epochs[-1]
                    if self.epochs - 50 <= int(last_saved_epoch):
                        rm_filename = [filename for filename in filenames if last_saved_epoch in filename][0]
                        os.remove(save_loc+rm_filename)
                    
                except IndexError:
                    pass

        # Get current time in YYMMDDhhmm format
        timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        
        # Get epoch_num as string
        epoch = str('%i'%self.epochs)
        
        # Get training_error as string
        loss = str(self.valid_loss_store[-1]).replace('.','-')
        
        output_filename = '%s_epoch_%s_loss_%s.pth'%(timestamp, epoch, loss)
        
        assert os.path.splitext(output_filename)[1] == '.pth', 'Output filename must have .pth extension'
                
        # Create dictionary of training variables
        train_dict = {'best' : self.best, 'train_loss_store': self.train_loss_store,
                      'valid_loss_store' : self.valid_loss_store,
                      'full_loss_store' : self.full_loss_store,
                      'epochs' : self.epochs, 'current_step' : self.current_step,
                      'last_decay_epoch' : self.last_decay_epoch,
                      'learning_rate' : self.learning_rate,
                      'cost_weights' : self.cost_weights,
                      'dataset_name' : self.dataset_name}
        
        # Save network parameters, optimizer state, and training variables
        torch.save({'net' : self.state_dict(), 'opt' : self.optimizer.state_dict(), 'train' : train_dict},
                   save_loc+output_filename)

    
    def load_checkpoint(self, input_filename='best', output='.'):
        '''
        Load checkpoint of network parameters and optimizer state

        required arguments:
            - input_filename (string): options:
                - path to input file. Must have .pth extension
                - 'best' (default) : checkpoint with lowest saved loss
                - 'recent'         : most recent checkpoint
                - 'longest'        : checkpoint after most training
            
            - dataset_name : if input_filename is not a path, must not be None
        '''
        save_loc = '%s/models_sim/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)
        
        # If input_filename is not a filename, get checkpoint with specified quality (best, recent, longest)
        if not os.path.exists(input_filename):
            # Get checkpoint filenames
            try:
                _,_,filenames = list(os.walk(save_loc))[0]
            except IndexError:
                return
            
            if len(filenames) > 0:
            
                # Sort in ascending order
                filenames.sort()

                # Split filenames into attributes (dates, epochs, loss)
                split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
                dates = [att[0] for att in split_filenames]
                epoch = [att[2] for att in split_filenames]
                loss  = [att[-1] for att in split_filenames]

                if input_filename == 'best':
                    # Get filename with lowest loss. If conflict, take most recent of subset.
                    loss.sort()
                    best = loss[0]
                    input_filename = [fn for fn in filenames if best in fn][-1]

                elif input_filename == 'recent':
                    # Get filename with most recent timestamp. If conflict, take first one
                    dates.sort()
                    recent = dates[-1]
                    input_filename = [fn for fn in filenames if recent in fn][0]

                elif input_filename == 'longest':
                    # Get filename with most number of epochs run. If conflict, take most recent of subset.
                    epoch.sort()
                    longest = epoch[-1]
                    input_filename = [fn for fn in filenames if longest in fn][-1]

                else:
                    assert False, 'input_filename must be a valid path, or one of \'best\', \'recent\', or \'longest\''
                    
            else:
                return

        assert os.path.splitext(input_filename)[1] == '.pth', 'Input filename must have .pth extension'

        # Load checkpoint
        state = torch.load(save_loc+input_filename)

        # Set network parameters
        self.load_state_dict(state['net'])

        # Set optimizer state
        self.optimizer.load_state_dict(state['opt'])

        # Set training variables
        self.best                  = state['train']['best']
        self.train_loss_store      = state['train']['train_loss_store']
        self.valid_loss_store      = state['train']['valid_loss_store']
        self.full_loss_store       = state['train']['full_loss_store']
        self.epochs                = state['train']['epochs']
        self.current_step          = state['train']['current_step']
        self.last_decay_epoch      = state['train']['last_decay_epoch']
        self.learning_rate         = state['train']['learning_rate']
        self.cost_weights          = state['train']['cost_weights']
        self.dataset_name          = state['train']['dataset_name']
        
    
    def health_check(self, writer):
        '''
        Checks the gradient norms for each parameter, what the maximum weight is in each weight matrix,
        and whether any weights have reached nan
        
        Report norm of each weight matrix
        Report norm of each layer activity
        Report norm of each Jacobian
        
        To report by batch. Look at data that is inducing the blow-ups.
        
        Create a -Nan report. What went wrong? Create file that shows data that preceded blow up, 
        and norm changes over epochs
        
        Theory 1: sparse activity in real data too difficult to encode
            - maybe, but not fixed by augmentation
            
        Theory 2: Edgeworth approximation ruining everything
            - probably: when switching to order=2 loss does not blow up, but validation error is huge
        '''
        
        hc_results = {'Weights' : {}, 'Gradients' : {}, 'Activity' : {}}
        odict = self._modules
        ii=1
        for name in odict.keys():
            if 'gru' in name:
                writer.add_scalar('3_Weight_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.data.norm(), self.current_step)
                writer.add_scalar('3_Weight_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.data.norm(), self.current_step)
                
                if self.current_step > 1:

                    writer.add_scalar('4_Gradient_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.grad.data.norm(), self.current_step)
                    writer.add_scalar('4_Gradient_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.grad.data.norm(), self.current_step)
            
            elif 'fc' in name or 'conv' in name:
                writer.add_scalar('3_Weight_norms/%i_%s'%(ii, name), odict.get(name).weight.data.norm(), self.current_step)
                if self.current_step > 1:
                    writer.add_scalar('4_Gradient_norms/%i_%s'%(ii, name), odict.get(name).weight.grad.data.norm(), self.current_step)
 
            ii+=1
        
        writer.add_scalar('5_Activity_norms/1_efgen', self.efcon.data.norm(), self.current_step)
        writer.add_scalar('5_Activity_norms/2_ebgen', self.ebcon.data.norm(), self.current_step)
        return
        
    
    def _set_params(self, params):
        for k in params.keys():
            self.__setattr__(k, params[k])

    
    def _update_params(self, prev_params, new_params):
        if new_params:
            params = update_param_dict(prev_params, new_params)
        else:
            params = prev_params
        self._set_params(params)