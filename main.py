import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
# from src.plotting import *
# from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from src.mtadgat_eval_methods import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import numpy as np


# from beepy import beep
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()
print()
print()


print(device)
print(args.sub_data)

def convert_to_windows(data, model):
    windows = [];
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)# + "_full")
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-2-1_' + file
        if dataset == 'SMAP': file = 'D-6_' + file
        # if dataset == 'MSL': file = 'MSL_' + file

        if dataset == 'MSL': file = 'T-5_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    #     ##-##
    #     f = open(os.path.join(folder, f'{file}.pkl'), "rb")
    #     loader.append(pickle.load(f).reshape((-1, 55)))
    #     f.close()
    #     ##-##
    
    # ##-##
    # f = open(os.path.join(folder, 'MSL_labels.pkl'), "rb")
    # loader.append(pickle.load(f).reshape((-1)))
    # f.close()
    # ##-##

    # loader = [i[:, debug:debug+1] for i in loader]
    # loader[0] = np.concatenate((loader[0], np.zeros((loader[0].shape[0], 4))), axis=1)
    # loader[1] = np.concatenate((loader[1], np.zeros((loader[1].shape[0], 4))), axis=1)
    # loader[2] = np.concatenate((loader[2], np.zeros((loader[2].shape[0], 4))), axis=1)
    
    # loader[0] = loader[0][17150:17400, :]
    # # print(loader[0].shape)
    # loader[1] = loader[1][17150:17400, :]
    # loader[2] = loader[2][17150:17400, :]
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    print("loader[0].shape[0]", str(loader[0].shape))
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    print("loader[1].shape[0]", str(loader[1].shape))
    labels = loader[2]

    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model_fd.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model_fd.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1] #number of features = 25
    if 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        print(data_x.shape)
        dataset = TensorDataset(data_x, data_x)
        
        bs = model.batch# if training else len(data)
        
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(device)
                # print("1", str(d.shape))
                # print('2', str(_.shape))
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                # print('w', str(window.shape))
                elem = window[-1, :, :].view(1, local_bs, feats) #[1,128,25]
                # print('elem', str(elem.shape))
                z = model(window, elem)
                # 128, 10, 25    128, 1, 25    -> 256, 100, 38  256, 1, 38
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            bada_z = [] ######
            bada_loss = [] ######

            for d, _ in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]

                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats) #############
                z = model(window, elem, training_flag=training)
                if isinstance(z, tuple): z = z[1]

                loss = l(z, elem)[0] #########
                bada_z.append(z.detach().cpu().numpy()[0]) #########
                bada_loss.append(loss.detach().cpu().numpy()) #########

            # loss = loss.to('cpu')
            # z = z.to('cpu')
            
            return np.concatenate(bada_loss, axis=0), np.concatenate(bada_z, axis=0) #########
            # return loss.detach().numpy(), z.detach().numpy()[0]

    # else:
    #     y_pred = model(data)
    #     loss = l(y_pred, data)
    #     if training:
    #         tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         return loss.item(), optimizer.param_groups[0]['lr']
    #     else:
    #         return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)
    # if args.model in ['MERLIN']:
    #     eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
    print("labels.shape", str(labels.shape))
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))

    # trainD = trainD[:128 * (trainD.shape[0] // 128), :]
    # testD = testD[:128*(testD.shape[0]//128), :]
    # labels = labels[:128 * (labels.shape[0] // 128), :]

    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                      'MAD_GAN'] or 'TranAD' in model.name:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)


    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5;
        e = epoch + 1;
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        # save_model(model, optimizer, scheduler, e, accuracy_list)
        # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')


    # attention_weights = []

    # def get_attention_hook(module, input, output):
    #     # Assuming output is the attention weights you're interested in
    #     attention_weights.append(output)

    # # Add the hook to the specific layer
    # for layer in model.transformer_encoder.layers:
    #     layer.self_attn.register_forward_hook(get_attention_hook)
    
    
    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    # print('hi')
    # with open("attention_weights3", "wb") as fp:   #Pickling
    #     pickle.dump(attention_weights, fp)

    # # print(attention_weights)
    # print('hi')

    np.save("preds.npy", y_pred)
    ### Plot curves
    # if not args.test:
    # 	if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
    # 	plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    # ### Scores
    # df = pd.DataFrame()
    # # lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    # for i in range(loss.shape[1]):

    #     l, ls = loss[:, i], labels[:, i]
    #     # lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        
    #     # print(lt, l, ls)
    #     # result, pred = pot_eval(lt, l, ls);
    #     result = bf_search(l, ls, start=0.01, end=2, step_num=100, verbose=False)
    #     # preds.append(pred)
    #     df = df.append(result, ignore_index=True)




    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    
    lossFinal = np.mean(loss, axis=1)
    # lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    print(2)
    # result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result = bf_search(lossFinal, labelsFinal, start=0.01, end=2, step_num=100, verbose=False)
    print(3)

    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    # print(df)
    # with pd.ExcelWriter('E:/tranad/TranAD/output.xlsx',mode='a') as writer:
    #     df.to_excel(writer, sheet_name='MBA_')

    pprint(result)

# pprint(getresults2(df, result))
# beep(4)
