import torch
import argparse
import random
from script.nets import *
from script.utils import *
from script.models import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-dp', '--data_path', default='PICL', type=str, help='data path name')
    parser.add_argument('-lf', '--logs_fname', default='nswe', type=str, help='logs file name')
    parser.add_argument('-dc', '--dict', default='nswe', type=str, help='dict name')
    parser.add_argument('-dr', '--data_rate', default=1, type=float, help='logs file name')
    
    parser.add_argument('-L', '--L', default=4, type=int, help='the number of layers')
    parser.add_argument('-m', '--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('-w', '--width', default=32, type=int, help='the number of width of FNO layer')
    parser.add_argument('--drop', default=0.1, type=float, help='Dropout of unet')
    parser.add_argument('-ch', '--channel', default=32, type=int, help='channels of unet')
    
    parser.add_argument('--phys_gap', default=40, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=10, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=2, type=int, help = 'Number of Phys Steps')
    parser.add_argument('--phys_scale', default=0.1, type=float, help = 'Number of Phys Scale')
    parser.add_argument('--phys_random_select', default=False, type=bool, help = 'Whether random select')

    parser.add_argument('--batch_size', default=32, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=3, type=int, help = 'Number of Epochs')
    parser.add_argument('--re_epochs', default=1, type=int, help = 'Number of Phys Retrain Epochs')
    parser.add_argument('--tune_epochs', default=1, type=int, help = 'Number of Data Retrain Epochs')
    parser.add_argument('--loop', default=2, type=int, help = 'Number of Total Loop')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--gpu', default=0, type=int, help='device number')

    parser.add_argument('-tg', '--tg', default=1, type=int, help = 'time gap')
    parser.add_argument('-Ng', '--Ng', default=1, type=int, help = 'N gap')
    parser.add_argument('-l1', '--lambda1', default=1, type=float, help='weight of data losses1')
    parser.add_argument('-l2', '--lambda2', default=0, type=float, help='weight of encode losses1')
    parser.add_argument('-l3', '--lambda3', default=0, type=float, help='weight of encode losses2')
    parser.add_argument('-l4', '--lambda4', default=100, type=float, help='weight of physics losses')
    parser.add_argument('-l5', '--lambda5', default=100, type=float, help='weight of physics losses of encoder')
    parser.add_argument('-l6', '--lambda6', default=0, type=float, help='weight of hidden loss')
    parser.add_argument('-fc', '--f_channels', default=0, type=int, help='channels of f encode')
    
    parser.add_argument('--re', default=0.1, type=float, help='re number')
    parser.add_argument('--lx', default=2.05, type=float, help='Lx')
    parser.add_argument('--ly', default=2.05, type=float, help='Ly')
    parser.add_argument('--en_dim', default=3, type=int, help='the input dimension of first layer of encoder')
    parser.add_argument('--gap_sample', default=5, type=int, help='sample gap from origin uvp in space')
    parser.add_argument('--obs_num', default=7, type=int, help='width and length of observation')
    parser.add_argument('--state_num', default=32, type=int, help='width and length of of state')
    parser.add_argument('--recall_size', default=4, type=int, help='recall history observation')

    parser.add_argument('--with_next', default=1, type=int, help='with ut+1 or not')
    parser.add_argument('--start_phys', default=500, type=int, help='with ut+1 or not')
    parser.add_argument('--seed', default=333, type=int, help='set random seed')
    parser.add_argument('--traj_str_tr', default=0, type=int, help='select the train set start')
    parser.add_argument('--traj_end_tr', default=400, type=int, help='select the train set end')
    parser.add_argument('--traj_str_ts', default=450, type=int, help='select the test set start')
    parser.add_argument('--traj_end_ts', default=500, type=int, help='select the test set end')

    parser.add_argument('-trsta', '--train_str', default=0, type=int, help='start position train')
    parser.add_argument('-trend', '--train_end', default=14, type=int, help='end position train')
    parser.add_argument('-tssta', '--test_str', default=0, type=int, help='start position test')
    parser.add_argument('-tsend', '--test_end', default=14, type=int, help='end position test')

    parser.add_argument('--unlabel_size', default=1/3, type=float, help='select the unlabel set scale')

    parser.add_argument('--ups', default=1, type=int, help='how to upsample in unet')
    parser.add_argument('--full', default=False, type=int, help='if the trans phys loss backward to encoder')

    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    args = get_args()
    print(args)

    # logs
    logs = dict()
    logs['args'] = args
    logs['mean'] = []
    logs['std'] = []
    logs['full'] = args.full
    
    num_obs = args.obs_num
    num_state = args.state_num
    gap = args.gap_sample
    trsta = args.train_str
    trend = args.train_end
    tssta = args.test_str
    tsend = args.test_end
    traj_str_tr = args.traj_str_tr
    traj_end_tr = args.traj_end_tr
    traj_str_ts = args.traj_str_ts
    traj_end_ts = args.traj_end_ts
    unlabel_size = args.unlabel_size
    tg = args.tg
    Ng = args.Ng 
    batch_size=args.batch_size
    dt = tg*0.01
    recall_size = args.recall_size
    seed = args.seed
    
    setup_seed(seed)
    
    logs_fname = f'results/{args.dict}/{args.logs_fname}'

    data_path = 'data/nswe_coarse'
    full_data_path = 'data/nswe_fine'
    
    data_run, data_test = data_import(data_path, full_data_path, traj_str_tr, traj_end_tr, traj_str_ts, traj_end_ts, 
                                      trsta, trend, tssta, tsend, num_obs, num_state, gap)
    
    ctr = torch.tensor(np.zeros((batch_size, data_run.shape[3], data_run.shape[4], 1)), dtype=torch.float64)  # (16,41,41,1)
    N0, nt, nx, ny = data_run.shape[0], data_run.shape[2], data_run.shape[3], data_run.shape[4]
    shape = [nx, ny]
    
    unlabel_data, unlabel_lab, train_data, train_labels, test_data, test_labels = data_divide(data_run, data_test, recall_size, N0, unlabel_size)

    train_dataset = MyDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = MyDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    retrain_dataset = MyDataset(unlabel_data, unlabel_lab)
    retrain_dataloader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    logs['pred_model'] = []
    logs['phys_inform_model'] = []
    
    logs['train_pred_loss'] = []
    logs['train_input_en_loss'] = []
    logs['train_output_en_loss'] = []
    logs['train_physics_loss'] = []
    logs['train_phys_encode_loss'] = []
    logs['train_hidden_loss'] = []
    logs['train_error_pred'] = []
    logs['train_error_input'] = []
    logs['train_error_output'] = []
    
    logs['test_pred_loss'] = []
    logs['test_input_en_loss'] = []
    logs['test_output_en_loss'] = []
    logs['test_physics_loss'] = []
    logs['test_phys_encode_loss'] = []
    logs['test_hidden_loss'] = []
    logs['test_error_pred'] = []
    logs['test_error_input'] = []
    logs['test_error_output'] = []

    logs['retrain_phys_loss'] = []
    logs['retrain_data_loss'] = []


    # model setting
    nswe_model = NSWEModel_FNO(shape, dt, args)
    params_num = nswe_model.count_params()

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nswe_model.device))
    print(f'param numbers of the model: {params_num}')

    # train process
    for loop in range(1,nswe_model.params.loop+1):
        for epoch in range(1, nswe_model.params.epochs+1):
            nswe_model.phys_inform(epoch, train_dataloader, ctr, logs)
            if epoch % 10 == 0:
                nswe_model.phys_test(test_dataloader, ctr, logs)
                
        for epoch in range(1, nswe_model.params.re_epochs+1):
            nswe_model.phys_retrain(epoch, retrain_dataloader, ctr, logs)
            if epoch % 10 == 0:
                nswe_model.phys_test(test_dataloader, ctr, logs)

        for epoch in range(1, nswe_model.params.tune_epochs+1):
            nswe_model.data_retrain(epoch, train_dataloader, ctr, logs)
            if epoch % 10 == 0:
                nswe_model.phys_test(test_dataloader, ctr, logs)

    torch.save([nswe_model.phys_inform_model.state_dict(), logs], logs_fname)