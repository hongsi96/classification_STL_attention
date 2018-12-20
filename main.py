import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from gradcam import *
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import os
import argparse
import numpy as np
from models import *
from utils import progress_bar
from PIL import Image
from tensorboardX import SummaryWriter
import pdb
parser = argparse.ArgumentParser(description='PyTorch STL10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=300, type=int, help='number of epoches')
parser.add_argument('--pre_train', default=None, type=str, help='address of pretrained model')
parser.add_argument('--save', default='EX', type=str, help='experiment repo')
parser.add_argument('--model', default='mobilev2', type=str, help='model name')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--attention', default=None, type=str, help='se | cbam | ge')
parser.add_argument('--test_only', action="store_true",  help='just execute test')

parser.add_argument('--optim', type=str, default='SGD',help='ADAM | SGD')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,help='ADAM epsilon for numerical stability')
#gradcam
parser.add_argument('--gradcam', action="store_true",  help='gradcam')
args = parser.parse_args()

writer = SummaryWriter(os.path.join('experiments',os.path.join(args.save,'runs')))


#params init
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_valid=0 # best valid accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_params=0


# Data
print('==> Preparing data..')
classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
train_loader = torch.utils.data.DataLoader(
    datasets.STL10(
        root='./data', split='test', download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

test_dataset=datasets.STL10(root='./data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

# create valid loader : 100 images for each class
sample_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,num_workers=2,shuffle=False)
valid_count=np.zeros(10)
valid_idx=[]
valid_each_class=100
for batch_idx, (inputs, targets) in enumerate(sample_loader):
    #print(targets)
    if valid_count[targets]<=valid_each_class-1:
        valid_idx.append(batch_idx)
        valid_count[targets]+=1
    if len(valid_idx)==valid_each_class*10:    
        break

num_test = len(test_dataset)
indices = list(range(num_test))
test_idx=[item for item in indices if item not in valid_idx]
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100,
        sampler=test_sampler,
        num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, sampler=valid_sampler,
        num_workers=2
)

#for batch_idx, (inputs, targets) in enumerate(valid_loader):
#    print(torch.sum(targets))
#pdb.set_trace()

# Model
print('==> Building model..')
if args.model =='mobilev2':
    net = MobileNetV2(args.attention)
elif args.model=='mnas':
    net = MnasNet(args.attention)
else:
    pdb.set_trace()
num_params= sum(p.numel() for p in net.parameters()) 


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.pre_train is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.pre_train)
    #pdb.set_trace()
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    num_params = checkpoint['num_params']

print('model size :')
print(num_params)

#pdb.set_trace()
if args.gradcam:
    grad_cam = GradCam(net)
criterion = nn.CrossEntropyLoss()
if args.optim=='ADAM':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,betas=args.betas, eps=args.epsilon,weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


# Training
def train(epoch):
    print('\nEpoch: %d' % int(epoch+1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar('Train/Acc', 100.*correct/total, epoch)

def gradcaM():
    
    correct = 0
    wrong = 0
    total_correct=10
    total_wrong=10
    #net.eval()
    directory='experiments/'+args.pre_train.split('/')[1]+'/gradcam'

    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+'/correct')
        os.makedirs(directory+'/wrong')

    for batch_idx, (inputs, targets) in enumerate(sample_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        _, predicted = outputs.max(1)

        if targets.item() ==predicted.item() and correct<total_correct:
            correct+=1
            feature_image = grad_cam(inputs).squeeze(dim=0)
            #pdb.set_trace()
            origin_image = transforms.ToPILImage()(denorm(inputs.squeeze(dim=0)).cpu())
            origin_image = origin_image.resize((300,300), Image.ANTIALIAS)
            origin_image.save(directory+'/correct'+'/{}_ori_{}.png'.format(correct,classes[targets.item()]))
            grad_image = transforms.ToPILImage()(feature_image.squeeze(dim=0))
            grad_image = grad_image.resize((300,300), Image.ANTIALIAS)
            grad_image.save(directory+'/correct'+'/{}_grad_{}.png'.format(correct,classes[predicted.item()]))

        elif targets.item() !=predicted.item() and wrong<total_wrong:
            wrong+=1
            feature_image = grad_cam(inputs).squeeze(dim=0)
            origin_image = transforms.ToPILImage()(denorm(inputs.squeeze(dim=0)).cpu())
            origin_image = origin_image.resize((300,300), Image.ANTIALIAS)
            origin_image.save(directory+'/wrong'+'/{}_ori_{}.png'.format(wrong,classes[targets.item()]))

            grad_image = transforms.ToPILImage()(feature_image.squeeze(dim=0))
            grad_image = grad_image.resize((300,300), Image.ANTIALIAS)
            grad_image.save(directory+'/wrong'+'/{}_grad_{}.png'.format(wrong,classes[predicted.item()]))

        if correct>=total_correct and wrong>=total_wrong:
            print('grad cam : done')
            break



def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Test_Loss: %.3f | Test_Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test/Acc : {}'.format(100.*correct/total))

def valid(epoch):
    global best_acc_valid
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valid_loader), 'Valid_Loss: %.3f | Valid_Acc: %.3f%% (%d/%d)'
                % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar('Valid/Acc',100.*correct/total, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_valid:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'num_params': num_params,
        }

        torch.save(state, 'experiments/'+args.save+'/valid_model.t7')
        best_acc_valid= acc

if args.pre_train is not None and args.test_only:
    test()

elif args.pre_train is not None and args.gradcam:
    gradcaM()
else:    
    for epoch in range(start_epoch, start_epoch+args.epoch):
        scheduler.step()
        train(epoch)
        valid(epoch)