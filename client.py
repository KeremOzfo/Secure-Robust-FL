from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *
from nn_classes import get_net
from opacus.data_loader import DPDataLoader
from opacus import GradSampleModule
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.accountants import RDPAccountant

class client():
    def __init__(self,id,dataset,device,args,**kwargs):
        self.id = id
        self.model = GradSampleModule(get_net(args)) ## OPACUS grad sampler
        self.args = args
        self.device = device
        self.loader = DataLoader(dataset,batch_size=args.bs,shuffle=True)
        self.poisson_loader = DPDataLoader.from_data_loader(self.loader) ## Opacus Poisson loader
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = None ## For all optims
        self.momentum2 = None ## Exclusive to adam and adamw
        self.step = 0 ## exclusive to adam and adamw optims
        self.local_steps = args.localIter ## number of local steps
        self.lr = args.lr ## learning rate
        self.mean_loss = None
        self.omniscient = False ## For byzantine clients
        self.relocate = False ## For byzantine clients
        self.opt_step = self.get_optim(args) # selected optimzer

    def local_step(self, batch): ## single minibatch step
        #self.model.register_forward_pre_hook(forbid_accumulation_hook)
        device = self.device
        self.model.to(device)
        x, y = batch
        x, y = x.to(device), y.to(device)
        self.model.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        
        if self.args.private_client_training:
            vec = self.grad_to_vec()
            vec = self.clip(vec) # grad clipping
            vec = self.privatize_grad(vec) # noise addition
            self.update_grad(vec) # put the privatized gradients to the model
        
        self.mean_loss = loss.item()
        self.opt_step() # update local model with custom optimizer
        self.model.to('cpu')

    def clip(self,grad):
        C = self.args.clip_val
        vec_norm = grad.norm(2, 1)
        multiplier = vec_norm.new(vec_norm.size()).fill_(1)
        multiplier[vec_norm.gt(C)] = C / vec_norm[vec_norm.gt(C)]
        grad *= multiplier.unsqueeze(1)
        return grad

    def grad_to_vec(self):
        res = []
        for p in self.model.parameters():
            res.append(p.grad_sample.view(p.grad_sample.size(0), -1))
        return torch.cat(res, dim=1).squeeze()

    def privatize_grad(self, grad, mechanism=None): ## custom nosie injection
        grad = grad.mean(0)
        grad += torch.randn_like(grad).to(self.device) * self.args.sigma # adding noise to the mean of the sample gradients

        return grad
    
    def update_grad(self,grad):
        self.model.zero_grad()
        for p in self.model.parameters():
            size = p.data.view(1,-1).size(1)
            p.grad = grad[:size].view_as(p.data).clone()
            grad = grad[size:]
        return

    def train_(self, embd_momentum=None): ## Traning
        iterator = iter(self.poisson_loader)
        flat_model = get_model_flattened(self.model, self.device)
        if embd_momentum is not None: ## Embedded momentum given by the PS (Scaffold, FedADC)
            self.momentum = torch.tensor(embd_momentum, device=self.device)
        elif self.momentum is None: ## generate new momentum at first comm round
            self.momentum = torch.tensor(torch.zeros_like(flat_model, device=self.device))
        self.momentum.to(self.device)

        for i in range(self.local_steps):
            batch = iterator.__next__()
            self.local_step(batch)
        self.momentum.to('cpu')

    def get_grad(self): ## get the last momentum value if momentum >0
        ## This function can be modified to return model difference between client and PS if needed.
        if self.args.opt == 'sgd':
            return torch.clone(self.momentum).detach()
        else: #for adams
            eps = 1e-08
            beta1, beta2 = self.args.betas
            new_moment = self.momentum.clone().detach() / (1- beta1**self.step)
            moment2 = self.momentum2.clone().detach() / (1- beta2 ** self.step)
            return new_moment / (torch.sqrt(moment2) + eps)

    def update_model(self, net_ps): ## sync local model with Global model
        pull_model(self.model, net_ps)

    def lr_step(self): ## custom lr schedular,
        self.lr *= .1

    def get_optim(self,args):
        if args.opt == 'sgd':
            return self.step_sgd
        elif args.opt == 'adam':
            return self.step_adam
        elif args.opt == 'adamw': # if local iter is 1, regularization has no impact
            return self.step_adamw
        else:
            raise NotImplementedError('Invalid optimiser name')

    def step_sgd(self,noise=0): ## custom SGD function
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)

                if self.momentum is None:
                    buf = torch.clone(d_p).detach()
                else:
                    length, dims = d_p.numel(), d_p.size()
                    buf = self.momentum[last_ind:last_ind + length].view(dims).detach()
                    buf.mul_(args.Lmomentum)
                    buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                    if not args.embd_momentum:
                        self.momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                    last_ind += length
                if args.nesterov:
                    d_p = d_p.add(buf, alpha=args.Lmomentum)
                else:
                    d_p = buf
                p.data.add_(d_p, alpha=-self.lr)

    def step_adam(self): ## custom ADAM optimizer
        last_ind = 0
        args = self.args
        eps = 1e-08
        self.step += 1
        if self.momentum2 is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.tensor(torch.zeros(model_size,device=self.device))
            self.momentum2 = torch.tensor(torch.zeros(model_size, device=self.device))
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas[0]) + d_p.mul(1-args.betas[0])
                v_t = buf2.mul(args.betas[1]) + torch.pow(d_p,2).mul(1-args.betas[1])
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - (args.betas[0]**self.step))
                vt_h = v_t.div(1 - (args.betas[1]**self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)


    def step_adamw(self): ## custom AdamW optimizer
        args = self.args
        last_ind = 0
        eps = 1e-08
        self.step += 1
        if self.momentum is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.tensor(torch.zeros(model_size, device=args.device))
            self.momentum2 = torch.tensor(torch.zeros(model_size, device=args.device))
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                p.data.add_(p.data, alpha=args.wd * -self.lr)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas(0)) + d_p.mul(1-args.betas(0))
                v_t = buf2.mul(args.betas(1)) + torch.pow(d_p,2).mul(1-args.betas(1))
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - torch.pow(args.betas(0), self.step))
                vt_h = v_t.div(1 - torch.pow(args.betas(1), self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)

