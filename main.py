import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
#from models.GGAT import GraphCnn
criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    if args.tqdm:
        pbar = tqdm(range(total_iters), unit='batch')
    else:
        pbar = range(total_iters)

    loss_accum = 0
    loss_att_sum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # output,loss_att = model(batch_graph)  # add supersived into attention weight
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        # loss = criterion(output, labels) - 10*loss_att   #spurvised loss for attention
        loss = criterion(output,labels)
        #print(loss_att)
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        #loss_att_sum += loss_att

        #report
        if args.tqdm:
            pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
   # avg_loss_att = loss_att_sum / total_iters
   #print("epoch:{},loss training:{},loss_att:{}".format(epoch,average_loss,avg_loss_att))
    print("epoch:{},loss train:{}".format(epoch,average_loss))
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        now_graphs=[graphs[j] for j in sampled_idx]
        res = model(now_graphs)
        output.append(res.detach())
       # output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def ftest(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()
    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    #val_loss = criterion(output, labels)
    print("accuracy train: %f test: %f " % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=500,   #defeat
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wl2', type=float, default=0.0,
                        help='learning rate (default: 0.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "meta-guass-pow10",
                                        help='output file')
    parser.add_argument('--attention', type=bool, default=True,  #defeault false
                       help='if attention,defeaut:False')
    parser.add_argument('--tqdm', type=bool, default=False,
                        help='if use tqdm')
    parser.add_argument('--multi_head', type=int, default=1,  # defeat:3
                        help='if use tqdm')
    parser.add_argument('--sum_flag', type=int, default=1,     #defeatut :1
                        help='if 0: don;t sum')
    parser.add_argument('--inter', type=int, default=1,        #defeult :0
                        help='if 0: not do unteraction in attention')

    parser.add_argument('--dire_sigmod', type=int, default=0,  # defeult :0
                        help='if 0: do softmax in dire attention,else if 1: sigmod')

    parser.add_argument('--attention_type', type=str, default="mlp-sigmod",
                        help='attention type:dire(sum or not), mlp-softmax , mlp-sigmod')
    args = parser.parse_args()

    ISOTIMEFORMAT = '%Y-%m-%d-%H-%M'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    save_fold_path = "result/save_model/" + args.filename+str(args.num_layers) + str(theTime)  # result save path

    writer_path = str(theTime)+args.filename + 'TBX'
    writer = SummaryWriter(log_dir=writer_path)

    #set up seeds and gpu device
    print("lr:",args.lr)
    print("attention: ",args.attention)
    print("attention_type:",args.attention_type)
    print("sum_flag:",args.sum_flag)
    print("inter:",args.inter)
    print("filename:",args.filename)
    if args.attention == True:   # if do attention we need sum  graph pool information
        args.graph_pooling_type = 'sum'
    print("data sets:",args.dataset)
    print("degree as tag:",args.degree_as_tag)
    print("flod_idx:",args.fold_idx)
    if args.sum_flag == 1:
        print("if use  directly attention is sum attention ,besides use sigmod attention model")

    f = open(args.filename+"_train", 'w')
    if args.fold_idx == -1:
        acc = []
        for idx in range(10):
            acc_i = cross_val(args, writer, idx, f)
            acc.append(acc_i)
        writer.close()
        np.save("result/"+args.filename+"_all.numpy",np.array(acc))   # save
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device,attention=args.attention,multi_head=args.multi_head,sum_flag=args.sum_flag,inter=args.inter,attention_type=args.attention_type,dire_sigmod=args.dire_sigmod).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wl2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        #args.epoch = 1
        acc = []
        max_acc = 0
        for epoch in range(1, args.epochs + 1):#args.epochs
            scheduler.step()
            avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
            acc_train, acc_test = ftest(args, model, device, train_graphs, test_graphs, epoch)
            max_acc =max(acc_test,max_acc)
            writer.add_scalars(str(args.fold_idx) + '/scalar/acc', {'train': acc_train, 'val': acc_test}, epoch)
            acc.append(acc_test)
            f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
            f.write("\n")
            print("")
            if epoch % 50 == 0:
                torch.save(model.state_dict(),save_fold_path+"_"+str(epoch)+".pt")
        print("****************max acc:",max_acc)
        try:
            torch.save(model.state_dict(),save_fold_path+"_last.pt")
            np.save("result/"+args.filename+"_"+str(args.fold_idx)+"_val_acc.npy",np.array(acc))
            writer.close()
        except:
            print("acc all:",acc)
            pass
        #print(model.eps)
        f.close()

def cross_val(args,writer,idx,f):
    fold_idx = idx
    print("**********************fold:{}**************************************************".format(fold_idx))
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed,fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                     num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                     args.neighbor_pooling_type, device, attention=args.attention,multi_head=args.multi_head,sum_flag=args.sum_flag,inter=args.inter,attention_type=args.attention_type,dire_sigmod=args.dire_sigmod).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    f.write("************************************** %d *********" % fold_idx)
    max_acc = 0
    acc = []
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        writer.add_scalars('/scalar/acc' + str(fold_idx), {'train': acc_train, 'val': acc_test}, epoch)
        acc.append(acc_test)
        if acc_test>max_acc:
            max_acc = acc_test

        f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
        f.write("\n")
        print("")
    print("acc:",acc)
    try:
        f.write("************************************** flod_id:{},best:{} *********".format(fold_idx,max_acc))
        np.save("result/"+args.filename+"_"+str(fold_idx)+"_val_acc.npy",np.array(acc))
        print("************************************** flod_id:{},best:{} *********".format(fold_idx,max_acc))
    except:
        pass
    return acc



if __name__ == '__main__':
    main()
