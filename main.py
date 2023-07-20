import numpy as np
import torch.cuda
from parameters import args_parser
from nn_classes import get_net
from torch.utils.data import DataLoader
from data_loader import get_dataset, get_indices, DatasetSplit
from mapper import *
from utils import *
import operator
import functools

def run(args,device):
    num_client = args.num_client
    num_byz = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    if args.traitor > 0:
        traitors = np.random.choice(range(num_client), num_byz, replace=False)
    else:
        traitors = []
    assert num_byz == len(traitors)
    loyal_clients,traitor_clients = [], []
    trainset, testset = get_dataset(args)
    total_sample = trainset.__len__()
    sample_inds, data_map = get_indices(trainset, args)
    print(device)
    net_ps = get_net(args).to(device)
    print('number of parameters ', round(count_parameters(net_ps) / 1e6,3) ,'M')
    testloader = DataLoader(testset,128,shuffle=False,pin_memory=True)
    aggr = get_aggr(args)
    epoch = 0
    lr = args.lr
    accs = []
    clip_norm,clips = [], []
    krum_bypass, bypasses = [], []
    ep_loss, losses = [], []
    robust_update = torch.zeros(count_parameters(net_ps),device=device)
    for i in range(num_client):
        worker_dataset = DatasetSplit(trainset,sample_inds[i])
        #worker_data_map = data_map[i]
        if i in traitors:
            traitor_clients.append(get_attacker_cl(i,worker_dataset,device,args))
        else:
            loyal_clients.append(get_benign_cl(i,worker_dataset,device,args))
    [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]]

    ########### Clusters are not employed right now !
    if args.Byz_each_cluster and num_byz>0:
        clusters = np.array_split(loyal_clients,args.num_clusters)
        clusters = [cluster.tolist() for cluster in clusters]
        print(clusters)
        for i in range(num_byz):
            cluster_ind = i % args.num_clusters
            clusters[cluster_ind].append(traitor_clients[i])
    else:
        all_clients = [*loyal_clients, *traitor_clients]
        np.random.shuffle(all_clients)
        clusters = np.array_split(all_clients,args.num_clusters)
        clusters = [cluster.tolist() for cluster in clusters]
    np.random.shuffle(clusters)


    while epoch < args.global_epoch:
        [cl.train_() for cl in loyal_clients]
        if num_byz >0:
            if traitor_clients[0].omniscient:
                if args.MITM:
                    benign_grads = [cl.get_grad() for cl in loyal_clients]
                else:
                    [cl.train_psuedo_moments() for cl in traitor_clients]
                    benign_grads = [cl.get_benign_preds() for cl in traitor_clients]
                    benign_grads = functools.reduce(operator.iconcat, benign_grads, [])
                    #real_grads = [cl.get_grad() for cl in loyal_clients]
                    #check_dist(real_grads,benign_grads,device)
                [cl.omniscient_callback(benign_grads) for cl in traitor_clients]
            else:
                [cl.train_() for cl in traitor_clients]
        outputs = [cl.get_grad() for cl in [*loyal_clients, *traitor_clients]]
        assert len(outputs) == num_client
        ep_loss.append(sum([cl.mean_loss for cl in loyal_clients]) / len(loyal_clients))
        if args.aggr == 'cc' and num_byz > 0:
            byz_out = outputs[-1]
            aggr_dif = byz_out.sub(robust_update)
            clip_byz = aggr.clip(aggr_dif)
            clip_norm.append((aggr_dif-clip_byz).norm().item())
        elif args.aggr == 'krum' and num_byz >0:
            krum_bypass.append(aggr.success)
        robust_update = aggr.__call__(outputs)
        ps_flat = get_model_flattened(net_ps, device)
        ps_flat.add_(robust_update, alpha=-lr)
        unflat_model(net_ps, ps_flat)
        prev_epoch = int(epoch)
        epoch += (num_client * args.localIter * args.bs) / total_sample
        current_epoch = int(epoch)
        [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]]
        if num_byz > 0:
            if traitor_clients[0].relocate:
                [cl.get_global_m(robust_update.clone()) for cl in traitor_clients]
        if current_epoch > prev_epoch:
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
            if current_epoch in args.lr_decay:
                [cl.lr_step()for cl in [*loyal_clients,*traitor_clients]]
                lr *= .1
    return accs,losses,clips,bypasses


if __name__ == '__main__':
    args = args_parser()
    if torch.cuda.is_available() and args.device != 'cpu':
        device = args.device
    else:
        print('No nvidia gpu is not available, selecting CPU as device')
        device = 'cpu'
    dims = (args.trials, args.global_epoch)
    accs_all, losses_all, clips_all = np.empty(dims), np.empty(dims), np.empty(dims)
    bypass_all = np.empty(dims)
    for i in range(args.trials):
        accs, losses, clips, bypass = run(args, device)
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
