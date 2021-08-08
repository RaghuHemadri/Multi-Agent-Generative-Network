from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
import datetime
from torchsummary import summary

from networks import weights_init, generator_actor, discriminator_actor, generator_critic, discriminator_critic, generator_new_state, init_hidden_state
from utils import randomize_policy, save_models, soft_update, get_celeba
from reply_buffer import fake_replay_buffer, real_replay_buffer

torch.backends.cudnn.benchmark = True

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize" : 16,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 50,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2,# Save step.
    'gamma': 0.95,# Discount Factor
    'LSTM_hidden': 128,
    'LSTM_layers': 2,
    'GC_hidden': 256,
    'Generator_Q': 100,
    'DC_hidden': 256,
    'Discriminator_Q': 100,
    'noise_rate': 0.1,# noise rate for sampling from a standard normal distribution
    'epsilon': 0.1,# Epsilon Greedy
    'high_action': 0.5,# Clipping value for policy
    'dataset_root': 'data'}

# Creating Model Directory
if not os.path.isdir('models'):
    os.mkdir('models')
model_dir = 'models/' + datetime.datetime.now().strftime('%d_%B_%Y;%H_%M_%S%P')
os.mkdir(model_dir)

# Save Parameters in the Model Directory
filehandler = open(model_dir + '/parameters.txt', 'wt')
data = str(params)[1:-1]
filehandler.write(data)
filehandler.close()

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
state_loader = get_celeba(params)
new_state_loader = get_celeba(params)
real_data_loader = get_celeba(params)

# Create Networks
gen_ac = generator_actor(params).to(device)
gen_cr = generator_critic(params).to(device)
dis_ac = discriminator_actor(params).to(device)
dis_cr = discriminator_critic(params).to(device)
gen_new_state = generator_new_state(params).to(device)

# Initialize the Weights
gen_ac.apply(weights_init)
gen_cr.apply(weights_init)
dis_ac.apply(weights_init)
dis_cr.apply(weights_init)
gen_new_state.apply(weights_init)

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizers
optim_gen_ac = optim.Adam(gen_ac.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optim_gen_cr = optim.Adam(gen_cr.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optim_dis_ac = optim.Adam(dis_ac.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optim_dis_cr = optim.Adam(dis_cr.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optim_gen_new_state = optim.Adam(gen_new_state.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Target Networks
target_gen_ac = generator_actor(params).to(device)
target_gen_cr = generator_critic(params).to(device)
target_dis_ac = discriminator_actor(params).to(device)
target_dis_cr = discriminator_critic(params).to(device)
#target_gen_new_state = generator_new_state()

# Initialize Target Networks
target_gen_ac.load_state_dict(gen_ac.state_dict())
target_gen_cr.load_state_dict(gen_cr.state_dict())
target_dis_ac.load_state_dict(dis_ac.state_dict())
target_dis_cr.load_state_dict(dis_cr.state_dict())
#target_gen_new_state.load_state_dict(gen_new_state.state_dict())

fixed_noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)
label = torch.full((params['bsize'], ), real_label, device=device)

print("Starting Training Loop...")
print("-"*25)

# Stores generated images as training progresses.
img_list = []
iters = 0

# Initialize Hidden State
h, c = init_hidden_state(params, device)

# print(summary(gen_ac, [(3, 64, 64), (2, 128), (2, 128)]))
# temp = torch.randn(128, 3, 64, 64, device=device)
# print(gen_ac(temp, h, c))

for epoch in range(params['nepochs']):
    for i, (c_state, n_state, real) in enumerate(zip(state_loader, new_state_loader, real_data_loader), 0):
        # Transfer Data to Device
        c_state_data = c_state[0].to(device)
        n_state_data = n_state[0].to(device)
        real_data = real[0].to(device)

        # Train Generator Agent
        policy, n_policy, h, c = gen_ac(c_state_data, h, c)

        random_policy = randomize_policy(policy, params, device)

        gn_state = gen_new_state(n_policy)

        with torch.no_grad():
            q_next = target_gen_cr(n_state_data, n_policy)

            dis_pred = dis_ac(gn_state).view(-1)
            r_gen = torch.log(dis_pred)

            gen_tar_q = (r_gen + params['gamma'] * q_next).detach()

        gen_q_value = gen_cr(c_state_data, random_policy)
        critic_loss = (gen_tar_q - gen_q_value).pow(2).mean()

        actor_loss = -(gen_cr(c_state_data, policy).mean() + gen_cr(n_state_data, n_policy).mean())

        optim_gen_ac.zero_grad()
        actor_loss.backward()
        optim_gen_ac.step()

        optim_gen_cr.zero_grad()
        critic_loss.backward()
        optim_gen_cr.step()

        # Train Discriminator Agent
        dis_policy_fake = dis_ac(gn_state)
        dis_policy_real = dis_ac(real_data)

        with torch.no_grad():
            new_dis_policy_real = target_dis_ac(n_state_data)
            _, n_state_policy = gen_ac(n_state_data)
            new_gn_state = gen_new_state(n_state_policy)
            new_dis_policy_fake = target_dis_ac(new_gn_state)
            
            q_next_real = target_dis_cr(n_state_data, new_dis_policy_real)
            q_next_fake = target_dis_cr(new_gn_state, new_dis_policy_fake)

            r_real = torch.log(dis_policy_real.view(-1))
            r_fake = torch.log(label-dis_policy_fake.view(-1))

            dis_tar_q_real = (r_real + params['gamma']*q_next_real).detach()
            dis_tar_q_fake = (r_fake + params['gamma']*q_next_fake).detach()

        dis_q_real = dis_cr(real_data, dis_policy_real)
        dis_q_fake = dis_cr(gn_state, dis_policy_fake)

        dis_critic_loss_real = (dis_tar_q_real - dis_q_real).pow(2).mean()
        dis_critic_loss_fake = (dis_tar_q_fake - dis_q_fake).pow(2).mean()

        dis_critic_loss = dis_critic_loss_real + dis_critic_loss_fake

        dis_actor_loss = -(dis_q_real.mean() + dis_q_fake.mean())

        dis_ac.zero_grad()
        dis_cr.zero_grad()
        gen_new_state.zero_grad()

        dis_actor_loss.backward()
        dis_critic_loss.backward()

        optim_dis_ac.step()
        optim_dis_cr.step()
        optim_gen_new_state.step()

        # Update Target Networks
        target_dis_ac = soft_update(target_dis_ac, dis_ac)
        target_dis_cr = soft_update(target_dis_cr, dis_cr)
        target_gen_ac = soft_update(target_gen_ac, gen_ac)
        target_gen_cr = soft_update(target_gen_cr, gen_cr)

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(state_loader)-1)):
            with torch.no_grad():
                fake_data = gen_new_state(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))
        iters += 1

    # Save the Models
    if epoch%params['save_epoch'] == 0:
        save_models(gen_ac, gen_cr, dis_ac, dis_cr, gen_new_state, optim_gen_ac, 
                    optim_gen_cr, optim_dis_ac, optim_dis_cr, optim_gen_new_state,
                    params, model_dir, epoch)

# Save Final Model
save_models(gen_ac, gen_cr, dis_ac, dis_cr, gen_new_state, optim_gen_ac, 
            optim_gen_cr, optim_dis_ac, optim_dis_cr, optim_gen_new_state,
            params, model_dir)

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('celeba.gif', dpi=80, writer='imagemagick')