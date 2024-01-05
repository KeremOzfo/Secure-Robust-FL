from parameters_th import args_parser
from nn_classes import get_net
from torch.utils.data import DataLoader
import data_loader as dl
from mapper import *
from utils import *
import argparse
from pynvml import *
from torch.multiprocessing import set_start_method
import multiprocessing as mp
import itertools
import torch.multiprocessing as mpcuda
import time
from clustering import Clusters
import operator
import functools

#######################
## This main uses parameters_threaded.py


def run(args,device):
    num_client = args.num_client
    num_byz = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    if args.traitor > 0:
        traitors = np.random.choice(range(num_client), num_byz, replace=False)
    else:
        traitors = []
    assert num_byz == len(traitors)
    loyal_clients,traitor_clients = [], []
    trainset, testset = dl.get_dataset(args)
    total_sample = trainset.__len__()
    sample_inds, data_map = dl.get_indices(trainset, args)
    net_ps = get_net(args).to(device)
    print('number of parameters ', round(count_parameters(net_ps) / 1e6,3) ,'M')
    testloader = DataLoader(testset,128,shuffle=False,pin_memory=True)
    mother_aggr = get_mother_aggr(args)
    epoch = 0
    lr = args.lr
    accs = []

    ##debug purposes
    clip_norm,clips = [], []
    krum_bypass, bypasses = [], []
    ep_loss, losses = [], []
    ##

    robust_update = torch.zeros(count_parameters(net_ps),device=device)
    for i in range(num_client):
        worker_dataset = dl.DatasetSplit(trainset,sample_inds[i])
        #worker_data_map = data_map[i]
        if i in traitors:
            traitor_clients.append(get_attacker_cl(i,worker_dataset,device,args))
        else:
            loyal_clients.append(get_benign_cl(i,worker_dataset,device,args))
    [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]]
    clusters = Clusters(args,loyal_clients,traitor_clients)
    while epoch < args.global_epoch:
        [cl.train_() for cl in loyal_clients]
        if num_byz >0:
            if traitor_clients[0].omniscient:
                if args.MITM: ## direclty use the updates of the benign clients
                    benign_grads = [cl.get_grad() for cl in loyal_clients]
                else: ## byzantines calculates and predict benign gradient by themselfs
                    [cl.train_psuedo_moments() for cl in traitor_clients]
                    benign_grads = [cl.get_benign_preds() for cl in traitor_clients]
                    benign_grads = functools.reduce(operator.iconcat, benign_grads, [])
                    #real_grads = [cl.get_grad() for cl in loyal_clients]
                    #check_dist(real_grads,benign_grads,device)
                [cl.omniscient_callback(benign_grads) for cl in traitor_clients]
            else:
                [cl.train_() for cl in traitor_clients] ## bit_fip and label_flip attacks
        clusters.shuffle()
        ep_loss.append(sum([cl.mean_loss for cl in loyal_clients]) / len(loyal_clients))

        ####
        if args.aggr == 'cc' and num_byz > 0: ## CC debugging
            byz_out = traitor_clients[-1].get_grad()
            aggr_dif = byz_out.sub(robust_update)
            clip_byz = mother_aggr.clip(aggr_dif)
            clip_norm.append((aggr_dif-clip_byz).norm().item())
        elif args.aggr == 'krum' and num_byz >0: ## krum debuggin
            krum_bypass.append(mother_aggr.success)

        outputs = clusters.aggr_clusters()
        robust_update = mother_aggr.__call__(outputs) ## aggregation
        ps_flat = get_model_flattened(net_ps, device)
        ps_flat.add_(robust_update, alpha=-lr) ## update global model
        unflat_model(net_ps, ps_flat) ## update global model
        prev_epoch = int(epoch)
        epoch += (num_client * args.localIter * args.bs) / total_sample
        current_epoch = int(epoch) ## update epoch
        [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]] ## broadcast the latest model
        if num_byz > 0:
            if traitor_clients[0].relocate:
                [cl.get_global_m(robust_update.clone()) for cl in traitor_clients] ## Byzantines estimate aggregated value

        ############
        if current_epoch > prev_epoch: ## Printing Information about the current epoch
            acc = evaluate_accuracy(net_ps, testloader, device)
            avg_clip = round(sum(clip_norm) / len(clip_norm),4) if len(clip_norm) >0 else None
            mean_train_loss = round(sum(ep_loss)/len(ep_loss),4)
            avg_krum_success = round(np.mean(krum_bypass),3) if len(krum_bypass) >0 else None
            debug_print = ''
            if args.aggr == 'cc':
                debug_print = 'clip_avg {}'.format(avg_clip)
            elif args.aggr == 'krum':
                debug_print = 'krum_bypassed {}'.format(avg_krum_success)
            print('Epoch',current_epoch,'Accuracy',round(acc,3) * 100,'|',
                  'Loss:',mean_train_loss,'|',
                  debug_print)
            accs.append(acc*100)
            clips.append(avg_clip)
            losses.append(mean_train_loss)
            bypasses.append(avg_krum_success)
            clip_norm = []
            ep_loss = []
            krum_bypass = []
        ##############
        if current_epoch in args.lr_decay:  # Update learning rate
            [cl.lr_step() for cl in [*loyal_clients, *traitor_clients]]
            lr *= .1
    return accs,losses,clips,bypasses


def main_thread(args):
    device = args.gpu_id if args.gpu_id > -1 else 'cpu'
    dims = (args.trials, args.global_epoch)
    accs_all, losses_all, clips_all = np.empty(dims), np.empty(dims), np.empty(dims)
    bypass_all = np.empty(dims)
    tm_bypass_all = np.empty(dims)
    for i in range(args.trials):
        accs, losses, clips, bypass= run(args, device)
        accs_all[i], losses_all[i] = accs, losses
        if clips[-1] is not None:
            clips_all[i] = clips
        if bypass[-1] is not None:
            bypass_all[i] = bypass
    if args.aggr == 'krum':
        save_results(args, Test_acc=accs_all, Train_losses=losses_all, Bypassed=bypass_all)
    elif args.aggr == 'cc':
        save_results(args, Test_acc=accs_all, Train_losses=losses_all, Clipping=clips_all)
    else:
        save_results(args, Test_acc=accs_all, Train_losses=losses_all)


if __name__ == '__main__':
    args = args_parser()
    worker_per_device = args.worker_per_device
    cuda = args.cuda
    cuda_info = None
    if torch.cuda.device_count() < 1:
        cuda = False
        print('No Nvidia gpu found to use cuda, overriding "cpu" as device')
    Process = mpcuda.Process if cuda else mp.Process
    available_gpus = torch.cuda.device_count() - len(args.excluded_gpus) if cuda else 0
    max_active_user = available_gpus * worker_per_device if cuda else worker_per_device
    first_gpu_share = np.repeat(worker_per_device, torch.cuda.device_count())
    first_gpu_share[args.excluded_gpus] = 0
    combinations = []
    work_load = []
    simulations = []
    w_parser = argparse.ArgumentParser()
    started = 0
    excluded_args = ['excluded_gpus','lr_decay']
    for arg in vars(args):
        arg_type = type(getattr(args, arg))
        if arg_type == list and arg not in excluded_args:
            work_ = [n for n in getattr(args, arg)]
            work_load.append(work_)
    for t in itertools.product(*work_load):
        combinations.append(t)
    print('Number of simulations is :',len(combinations))
    for combination in combinations:
        w_parser = argparse.ArgumentParser()
        listC = 0
        for arg in vars(args):
            arg_type = type(getattr(args, arg))
            if arg_type == list and arg not in excluded_args:
                new_type = type(combination[listC])
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=combination[listC], help='')
                listC += 1
            else:
                val = getattr(args, arg)
                new_type = type(getattr(args, arg))
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=val, help='')

        if cuda:
            if started < max_active_user:
                selected_gpu = np.argmax(first_gpu_share)
                first_gpu_share[selected_gpu] -= 1
            else:
                nvmlInit()
                cuda_info = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
                             for i in range(torch.cuda.device_count())]
                cuda_memory = np.zeros(torch.cuda.device_count())
                for i, gpu in enumerate(cuda_info):
                    if i not in args.excluded_gpus:
                        cuda_memory[i] = gpu.free
                selected_gpu = np.argmax(cuda_memory)
            print('Process {} assigned with gpu:{}'.format(started, selected_gpu))
            w_parser.add_argument('--gpu_id', type=int, default=selected_gpu,
                                  help='cuda device selected')  # assign gpu for the work
        else:
            w_parser.add_argument('--gpu_id', type=int, default=-1, help='cpu selected')  # assign gpu for the work

        w_args = w_parser.parse_args()
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        process = Process(target=main_thread, args=(w_args,))
        process.start()
        simulations.append(process)
        started += 1

        while not len(simulations) < max_active_user:
            for i, process_data in enumerate(simulations):
                if not process_data.is_alive():
                    # remove from processes
                    p = simulations.pop(i)
                    del p
                    time.sleep(10)
