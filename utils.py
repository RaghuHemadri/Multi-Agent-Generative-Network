import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        return target

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        return target

def save_models(gen_act, gen_ctc, dis_act, dis_ctc, 
                gen_news, gen_act_opt, gen_ctc_opt,
                dis_act_opt, dis_ctc_opt, gen_news_opt,
                params, model_dir, epoch = None):
    if not epoch:
        torch.save({
                    'generator_actor' : gen_act.state_dict(),
                    'generator_critic' : gen_ctc.state_dict(),
                    'discriminator_actor' : dis_act.state_dict(),
                    'discriminator_critic' : dis_ctc.state_dict(),
                    'generator_new_state' : gen_news.state_dict(),
                    'generator_actor_optimizer' : gen_act_opt.state_dict(),
                    'generator_critic_optimizer' : gen_ctc_opt.state_dict(),
                    'discriminator_actor_optimizer' : dis_act_opt.state_dict(),
                    'discriminator_critic_optimizer' : dis_ctc_opt.state_dict(),
                    'generator_new_state_optimizer' : gen_news_opt.state_dict(),
                    'params' : params
                    }, 'model/model_dir/model_epoch_{}.pth'.format(epoch))

    else:
        torch.save({
                'generator_actor' : gen_act.state_dict(),
                'generator_critic' : gen_ctc.state_dict(),
                'discriminator_actor' : dis_act.state_dict(),
                'discriminator_critic' : dis_ctc.state_dict(),
                'generator_new_state' : gen_news.state_dict(),
                'generator_actor_optimizer' : gen_act_opt.state_dict(),
                'generator_critic_optimizer' : gen_ctc_opt.state_dict(),
                'discriminator_actor_optimizer' : dis_act_opt.state_dict(),
                'discriminator_critic_optimizer' : dis_ctc_opt.state_dict(),
                'generator_new_state_optimizer' : gen_news_opt.state_dict(),
                'params' : params
                }, 'model/model_dir/model_final.pth')

# def load_model():
# def load_generator_only():

def get_celeba(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = dset.ImageFolder(root=params['dataset_root'], transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader

def randomize_policy(pi, params, device):
    noise_rate = params['noise_rate']
    epsilon = params['epsilon']
    high_action = params['high_action']
    pi.detach()
    u = pi.cpu().detach().numpy()
    if np.random.uniform() < epsilon:
        u = np.random.uniform(-high_action, high_action, u.shape)
    else:
        noise = noise_rate * high_action * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -high_action, high_action)
    return torch.from_numpy(u).to(device)