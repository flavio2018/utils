"""Utilities to train recurrent neural networks on variable-length sequences with PyTorch"""


import torch


def get_mask(lens, device='cpu'):
       """Given a list of lengths of sequences, create a tensor mask that signals
       the sequences that reached padded positions."""
       return torch.tensor([1 if l > 0 else 0 for l in lens], device=device)


def reduce_lens(lens):
	"""Given a list of lengths of sequences in a batch, reduce each length
	up to zero to keep track of valid and padded positions."""
	return [l-1 if l > 0 else l for l in lens]


def save_states(model, h_dict, c_dict, samples_len):
	"""Util used to save states of a recurrent model corresponding to the last
	element of each sequence in a batch."""
	target_states = torch.argwhere(get_mask(samples_len) == 0)
	for state in target_states:
		h_dict[1].setdefault(state.item(), model.h_t_1[state, :])
		c_dict[1].setdefault(state.item(), model.c_t_1[state, :])
		try:
			h_dict[2].setdefault(state.item(), model.h_t_2[state, :])
			c_dict[2].setdefault(state.item(), model.c_t_2[state, :])
		except AttributeError:
			pass
	return h_dict, c_dict


def save_states_dntm(model, h_dict, samples_len):
	target_states = torch.argwhere(get_mask(samples_len) == 0)
	for state in target_states:
		h_dict.setdefault(state.item(), model.controller_hidden_state[:, state])  # transposed shape
	return h_dict
