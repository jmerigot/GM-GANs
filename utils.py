import torch
import torchvision
import os
import json


def D_train(x, L_G, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output
    D_real_acc = ((D_output.squeeze() > 0.5) == y_real.squeeze()).float().mean().item()

    # train discriminator on fake
    # z = torch.randn(x.shape[0], 100).cuda()
    z = L_G(batch_size=x.shape[0]).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output
    
    D_fake_acc = ((D_output.squeeze() > 0.5) == y_fake.squeeze()).float().mean().item()

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    D_metrics = {
        "D_real_acc" : D_real_acc,
        "D_fake_acc" : D_fake_acc,
        "D_real_loss" : D_real_loss.detach().cpu().item(),
        "D_fake_loss" : D_fake_loss.detach().cpu().item()
    }
        
    return D_metrics


def G_train(x, L_G, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#


    G.zero_grad()

    z = L_G(batch_size=x.shape[0]).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    G_acc = ((D_output.squeeze() > 0.5) == y.squeeze()).float().mean().item()
    
    G_metrics = {
        "G_acc" : G_acc,
        "G_loss" : G_loss.detach().cpu().item()
    }
        
    return G_metrics



def save_models(L_G, G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))
    torch.save(L_G.state_dict(), os.path.join(folder,'L_G.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def generate_sample(L_G, G, epoch, config):
    
    batch_size = config["training"]["batch_size"]
    run_name = config["run_name"]
    dir = 'runs/' + run_name + f"/epoch_{epoch}"
    os.makedirs(dir, exist_ok=True)

    n_samples = 0
    while n_samples<20:
        # z = torch.randn(args.batch_size, 100).cuda()   
        z = L_G(batch_size=batch_size).cuda()
        x = G(z)
        x = x.reshape(batch_size, 28, 28)
        for k in range(x.shape[0]):
            if n_samples<20:
                torchvision.utils.save_image(x[k:k+1], os.path.join(dir, f'{n_samples}.png'))         
                n_samples += 1