from Aggregators.clipping import Clipping
from Aggregators.cm import CM
from Aggregators.krum import Krum
from Aggregators.rfa import RFA
from Aggregators.trimmed_mean import TM
from Aggregators.fedavg import fedAVG
from Aggregators.cc_seq import Clipping_seq
from Aggregators.bulyan import Bulyan
from Attacks.alie import alie
from Attacks.ipm import IPMAttack
from Attacks.rop import reloc
from Attacks.bit_flip import pgd_traitor
from Attacks.label_flip import label_flip_traitor
from Attacks.cw import cw_traitor
from client import client as loyal_client

aggr_mapper = {'cc': Clipping, 'cm': CM, 'krum': Krum, 'rfa': RFA, 'tm': TM,'avg':fedAVG,
               'scc':Clipping_seq,'bulyan':Bulyan}
attack_mapper ={'bit_flip':pgd_traitor,'label_flip':label_flip_traitor,
                'cw':cw_traitor,'alie':alie,'rop':reloc,
                'ipm':IPMAttack}



def get_aggr(args):
    alg = args.aggr
    num_client = args.num_client
    b= int(num_client * args.traitor)
    n= num_client-b-2
    aggr_params = {'cc': [args.tau],'cca':[args.tau], 'scc':[args.tau,args.buck_len,args.buck_avg,args.multi_clip],
        'cm': [None],'krum': [num_client,b,n], 'rfa': [3], 'tm': [b],'avg':[None],'bulyan':[num_client,b]}
    return aggr_mapper[alg](*aggr_params[alg])


def get_attacker_cl(id,dataset,device,args):
    num_client = args.num_client
    num_traitor = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    client_params = {'id':id,'dataset':dataset,'device':device,'args':args}
    attacker_params = {'n':num_client,'m':num_traitor,'z':args.z_max,'eps':args.epsilon}
    traitor_client = attack_mapper[args.attack](**attacker_params,**client_params)
    return traitor_client

def get_benign_cl(id,dataset,device,args):
    client_params = {'id': id, 'dataset': dataset, 'device': device, 'args': args}
    return loyal_client(**client_params)