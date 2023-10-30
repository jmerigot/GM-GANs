import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim



from model import Generator, Discriminator, Latent_Generator
from utils import D_train, G_train, save_models, load_config, generate_sample

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()
    

    config = load_config("config.json")

    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    writer = SummaryWriter("runs/" + config["run_name"])

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()
    L_G = Latent_Generator(config)


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 

    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)


    list_G_loss = []
    list_G_acc = []
    list_D_fake_loss = []
    list_D_real_loss = []
    list_D_fake_acc = []
    list_D_real_acc = []
    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):  
        
        epoch_G_loss = 0
        epoch_G_acc = 0
        epoch_D_real_loss = 0
        epoch_D_fake_loss = 0
        epoch_D_real_acc = 0
        epoch_D_fake_acc = 0

             
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_metrics = D_train(x, L_G, G, D, D_optimizer, criterion)
            G_metrics = G_train(x, L_G, G, D, G_optimizer, criterion)
            
            ### Update Metrics ###
            epoch_G_loss += G_metrics["G_loss"]
            epoch_G_acc += G_metrics["G_acc"]
            epoch_D_real_loss += D_metrics["D_real_loss"]
            epoch_D_fake_loss += D_metrics["D_fake_loss"]
            epoch_D_real_acc += D_metrics["D_real_acc"]
            epoch_D_fake_acc += D_metrics["D_fake_acc"]
        
        list_G_loss.append(epoch_G_loss / len(train_loader))
        list_G_acc.append(epoch_G_acc / len(train_loader))
        list_D_fake_acc.append(epoch_D_fake_acc / len(train_loader))
        list_D_real_acc.append(epoch_D_real_acc / len(train_loader))
        list_D_real_loss.append(epoch_D_real_loss / len(train_loader))
        list_D_fake_loss.append(epoch_D_fake_loss / len(train_loader))

        print(
            epoch,
            "\n\t - G loss: {:.4f}".format(list_G_loss[-1]),
            "G acc: {:.2%}".format(list_G_acc[-1]), 
            "D fake loss: {:.4f}".format(list_D_fake_loss[-1]),
            "D real loss: {:.4f}".format(list_D_real_loss[-1]),
            "D fake acc: {:.2%}".format(list_D_fake_acc[-1]),
            "D real acc: {:.2%}".format(list_D_real_acc[-1]),
        )
        
        writer.add_scalar('Generator/Loss', list_G_loss[-1], epoch)
        writer.add_scalar('Generator/Accuracy', list_G_acc[-1], epoch)
        writer.add_scalar('Discriminator/Loss/Fake', list_D_fake_loss[-1], epoch)
        writer.add_scalar('Discriminator/Loss/Real', list_D_real_loss[-1], epoch)
        writer.add_scalar('Discriminator/Accuracy/Fake', list_D_fake_acc[-1], epoch)
        writer.add_scalar('Discriminator/Accuracy/Real', list_D_real_acc[-1], epoch)

        generate_sample(L_G, G, epoch, config)
        
        if epoch % 10 == 0:
            save_models(L_G, G, D, 'checkpoints')
    writer.close()
    print('Training done')

        