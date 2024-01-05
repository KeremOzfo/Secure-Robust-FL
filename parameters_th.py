import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # technical params
    parser.add_argument('--trials', type=int, default=3, help='number of trials')
    parser.add_argument('--cuda', type=bool, default=True, help='Use cuda as device')
    parser.add_argument('--worker_per_device', type=int, default=5, help='parallel processes per device')
    parser.add_argument('--excluded_gpus', type=list, default=[], help='bypassed gpus')

    # Federated params
    parser.add_argument('--global_epoch', type=int, default=100, help='total cumulative epoch')
    parser.add_argument('--localIter', type=int, default=1, help='Local Epoch')
    parser.add_argument('--num_client', type=int, default=25, help='number of clients')
    parser.add_argument('--num_clusters', type=list, default=[3,4,5,7], help='number of clusters')
    parser.add_argument('--Byz_each_cluster', type=bool, default=False, help='Ensures that at least 1 Byzantine present in each cluster')
    parser.add_argument('--traitor', type=float, default=0, help='traitor ratio')
    parser.add_argument('--attack', type=str, default='bit_flip', help='see Attacks')
    parser.add_argument('--aggr', type=str, default='avg', help='robust Aggregators')
    parser.add_argument('--embd_momentum', type=bool, default=False, help='FedADC embedded momentum')
    parser.add_argument('--early_stop', type=bool, default=False, help='Early stop function')
    parser.add_argument('--MITM', type=bool, default=True, help='Adversary capable of man-in-middle-attack')

    # Privacy params
    parser.add_argument('--private_client_training', type=bool,default=True, help='if (loyal) clients train privately or not')
    parser.add_argument('--noise_from_cluster', type=bool, default=True,
                        help='noise added in the clusters')

    #parser.add_argument('--private_client_training', type=bool, default=True, help='if (loyal) clients train privately or not')
    parser.add_argument('--clip_val', type=list, default=[1,10.], help='norm bound for grads')
    parser.add_argument('--sigma', type=list, default=[.01,.1,1], help='noise std for privacy')



    # Defence params
    parser.add_argument('--tau', type=float, default=10, help='tau value for cc aggr')
    parser.add_argument('--buck_len', type=int, default=3, help='bucket length for sequential cc')
    parser.add_argument('--buck_avg', type=bool, default=True, help='average the bucket for sequential cc')
    parser.add_argument('--multi_clip', type=bool, default=False, help='Additional reference point for the s-CC aggregator')

    # attack params
    parser.add_argument('--z_max', type=float, default=1., help='attack scale,None for auto generate')
    parser.add_argument('--nestrov_attack', type=bool, default=False, help='clean step first')
    parser.add_argument('--epsilon', type=float, default=0.2, help='ipm attack scale')

    # RoP spesific parameters
    parser.add_argument('--pi', type=float, default=1, help='location of the attack,1 for full relocation to aggr reference')
    parser.add_argument('--angle', type=int, default=90, help='angle of the pert,90 is default')
    parser.add_argument('--lamb', type=float, default=.9, help='refence point for attack direction',)


    # optimiser related
    parser.add_argument('--opt', type=str, default='sgd', help='name of the optimiser')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--lr_decay', type=float, default=[75], help='lr drop at given epoch')
    parser.add_argument('--wd', type=float, default=0, help='weight decay Value')
    parser.add_argument('--Lmomentum', type=float, default=0.9, help='Local Momentum for SGD')
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999), help='betas for adam and adamw opts')
    parser.add_argument('--worker_momentum', type=bool, default=True, help='adam like gradiant multiplier for SGD (1-Lmomentum)')
    parser.add_argument('--nesterov', type=bool, default=False, help='nestrov momentum for Local SGD steps')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='mnist', help='see data_loader.py')
    parser.add_argument('--dataset_dist', type=str, default='dirichlet',
                        help='distribution of dataset; iid or sort_part, dirichlet')
    parser.add_argument('--numb_cls_usr', type=int, default=2,
                        help='number of label type per client if sort_part selected')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha constant for dirichlet dataset_dist,lower for more skewness')
    parser.add_argument('--bs', type=int, default=32, help='batchsize')

    # nn related
    parser.add_argument('--nn_name', type=list, default=['mlp_small','mnistnet'], help='simplecnn,simplecifar,VGGs resnet(8-9-18-20)')
    parser.add_argument('--weight_init', type=str, default='-',
                        help='nn weight init, kn (Kaiming normal) or - (None)')
    parser.add_argument('--norm_type', type=str, default='gn',
                        help='gn (GroupNorm), bn (BatchNorm), - (None)')
    parser.add_argument('--num_groups', type=int, default=32,
                        help='number of groups if GroupNorm selected as norm_type, 1 for LayerNorm')

    args = parser.parse_args()
    return args
