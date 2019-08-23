import torch
import torchvision
import torchvision.transforms as transforms


# import matplotlib.pyplot as plt
import numpy as np
import time

import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from probprec import Preconditioner

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(
    description="Run SGD on a tfobs test problem.")
# parser.add_argument("test_problem",
#                     help="Name of the test_problem (e.g. 'cifar10.cifar10_3c3d'")
# parser.add_argument("--data_dir",
#                     help="Path to the base data dir. If not set, tfobs uses its default.")
parser.add_argument("-bs", "-batch_size", required=True, type=int,
                    help="The batch size (positive integer).")
parser.add_argument("-wd", "-weight_decay", type=float,default=0.0,
                    help="Factor used for the weight_deacy.")

parser.add_argument("-nw", "-number_of_workers", type=int,default=2,
                    help="Number of Workers.")
# Learning rate, either constant or schedule
parser.add_argument("-lr","-learning_rate", required=True, type=float,
                    help="Initial learning rate (positive float) to use. To set a learning rate *schedule*, use '--lr_sched_epochs' and '--lr_sched_values' additionally.")
parser.add_argument("-N", "-num_epochs", required=True, type=int,
                    help="Total number of training epochs.")

parser.add_argument("-ei", "-evaluation_iteration", type=int,default=100,
                    help="Total number of training epochs.")

parser.add_argument("-po", "-prior_observations", type=int,default=10,
                    help="Number of observations to estimate prior hyperparameters.")
parser.add_argument("-nl", "-likelihoods", type=int,default=5,
                    help="Number of observations to estimate posterior.")
parser.add_argument("-pr", "-preconditioner_rank", type=int,default=2,
                    help="Rank of preconditioner.")



args = parser.parse_args()
print(args)




BATCH_SIZE=args.bs#64
NUM_WORKERS=args.nw#2
LEARNING_RATE=args.lr#1e-3
EVALUATION_ITERATION=args.ei#200
NUM_EPOCHS=args.N#5
WEIGHT_DECAY=args.wd#2e-3
est_rank=args.pr
est_prior=args.po
gather_obs=args.nl


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device('cpu')
torch.set_default_dtype(torch.float)

torch.manual_seed(42)
# device=torch.device('cpu')
# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data_deepobs/pytorch', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='./data_deepobs/pytorch', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def name(self):
            return "2conv_3dense"


    class Net3c3d(nn.Module):

        def __init__(self, num_classes=10):
            super(Net3c3d, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5,padding=0),#, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
                nn.Conv2d(64, 96, kernel_size=3,padding=0),#, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
                nn.Conv2d(96, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 3 * 3, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
            )
            # init the layers
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_normal_(module.weight)

                if isinstance(module, nn.Linear):
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)

        def forward(self, x):
            x = self.features(x)
    #         print(x.size())
            x = x.view(x.size(0), 128 * 3 * 3)
            x = self.classifier(x)
            return x

        def name(self):
            return "3conv_3dense"


    model = Net3c3d()


    model.to(device)


    # EVALUATION_ITERATION=500


    criterion = nn.CrossEntropyLoss()
    alphas=[]
    train_loss=[]
    test_loss=[]
    test_acc=[]
    alphas.append(LEARNING_RATE)

    # specify the optimizer class
    optimizer_class = optim.SGD

    # and its hyperparameters
    hyperparams = {} #'lr': 0.1} #'momentum': 0.99}
    Poptimizer = Preconditioner([{"params": model.features.parameters()},
                                 {"params": model.classifier.parameters()}],
                                est_rank=est_rank, num_observations=gather_obs, prior_iterations=est_prior,
                                optim_class=optimizer_class, **hyperparams)

    # Optimizer = optim.SGD([{"params": model.features.parameters()},
    #                        {"params": model.classifier.parameters()}],
    #                       lr=0.01)

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        if epoch > 0:
            Poptimizer.start_estimate()

        start_time_epoch=time.perf_counter()
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            Poptimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward(create_graph = True)
            Poptimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % EVALUATION_ITERATION == EVALUATION_ITERATION-1:    # print every [ei] mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / EVALUATION_ITERATION))

                train_loss.append(running_loss / EVALUATION_ITERATION)
                running_loss = 0.0


        epoch_time=time.perf_counter()-start_time_epoch
        #Evaluate on test set
        running_test_loss=0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for j, data in enumerate(testloader,0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_test_loss += loss.item()

         # print('epoch $d time %1.3e testloss %1.4e testacc %1.3e'%(epoch+1,time_epoch,))
        # print(total,correct,running_test_loss)
        test_acc.append(100.0*correct/total)
        test_loss.append(running_test_loss/j)
        print('epoch %d:  testloss %1.4e testacc %1.3e'%(epoch, test_loss[-1],test_acc[-1]))


    print('Finished Training')
    # print(train_loss)
    print(alphas)

    save_name='results/'
    save_name+='%s_psgd_%d_%d_%.4f_%.4f'%(model.name(),NUM_EPOCHS,BATCH_SIZE,LEARNING_RATE,WEIGHT_DECAY)
    save_name += '_' + time.strftime("%Y-%m-%d-%H-%M-%S")
    save_name += '_sqrt'

    save_data_train=[np.asarray(train_loss)]
    save_data_test=[np.asarray(test_loss),np.asarray(test_acc)]

# np.savetxt(save_name+'_train.txt', save_data_train, fmt='%1.5e', delimiter=' ')
# np.savetxt(save_name+'_test.txt', save_data_test, fmt='%1.5e', delimiter=' ')
# np.savetxt(save_name+'_alphas.txt',alphas, fmt='%1.5e', delimiter=' ')
