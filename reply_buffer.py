import numpy as np

class fake_replay_buffer:
	def __init__(self, batch_size, num_agents, gen_dim, 
				 dis_dim, image_size, image_channels):
		self.batch_size = batch_size
		self.num_agents = num_agents
		self.gen_dim = gen_dim
		self.dis_dim = dis_dim
		self.image_size = image_size
		self.image_channels = image_channels

		self.curr_state = np.zeros((self.batch_size, self.image_size,
									self.image_size, self.image_channels))
		self.new_state = np.zeros((self.batch_size, self.image_size,
									self.image_size, self.image_channels))
		self.reward = np.zeros((self.batch_size, self.num_agents))
		self.gen_policy = np.zeros((self.batch_size, self.gen_dim))
		self.dis_policy = np.zeros((self.batch_size, self.dis_dim))

	def store_transition(self, curr_state, new_state, reward,
						 policy, is_generator = True):
		self.curr_state = curr_state
		self.new_state = new_state
		if is_generator:
			self.reward[:][0] = reward
			self.gen_policy = policy
		else:
			self.reward[:][1] = reward
			self.dis_policy = policy

	def get_buffer(self):
		return [self.curr_state, self.new_state, self.reward, 
				self.gen_policy, self.dis_policy]

class real_replay_buffer:
	def __init__(self, batch_size, gen_dim, dis_dim, 
				 image_size, image_channels):
		self.batch_size = batch_size
		self.gen_dim = gen_dim
		self.dis_dim = dis_dim
		self.image_size = image_size
		self.image_channels = image_channels

		self.curr_state = np.zeros((self.batch_size, self.image_size,
									self.image_size, self.image_channels))
		self.new_state = np.zeros((self.batch_size, self.image_size,
									self.image_size, self.image_channels))
		self.reward = np.zeros((self.batch_size))
		self.gen_policy = np.zeros((self.batch_size, self.gen_dim))
		self.dis_policy = np.zeros((self.batch_size, self.dis_dim))

	def store_transition(self, curr_state, new_state, reward,
						 policy, is_generator = True):
		self.curr_state = curr_state
		self.new_state = new_state
		self.reward[:] = reward
		if is_generator:
			self.gen_policy = policy
		else:
			self.dis_policy = policy

	def get_buffer(self):
		return [self.curr_state, self.new_state, self.reward, 
				self.gen_policy, self.dis_policy]