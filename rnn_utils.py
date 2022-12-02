"""Utilities to train recurrent neural networks on variable-length sequences with PyTorch"""


import torch


_EOS = '.'
_SOS = '?'
_PAD = '#'


def make_1hot(char, token2pos, vocab_size):
	"""Given a char, vocabulary size and a dictionary associating each
	char to an index, builds a 1-hot representation of the char."""
	onehot_vec = torch.zeros((1, vocab_size))
	onehot_vec[0, token2pos[char]] = 1
	return onehot_vec


def make_tensor(sample, token2pos, vocab_size):
	"""Given a sequence, vocabulary size and a dictionary associating each
	char to an index, builds a tensor representation of the sequence."""
	sample_tensor = []
	for c in sample:
		sample_tensor.append(make_1hot(c, token2pos, vocab_size))
	return torch.concat(sample_tensor).unsqueeze(dim=0)


def make_target_tensor(target, token2pos, vocab_size):
    """Given a target sequence, vocabulary size and a dictionary associating each
    char to an index, builds a tensor representation of the target sequence."""
    target_tensor = []
    target_tensor.append(make_1hot(_SOS, token2pos, vocab_size))
    for c in target:
        target_tensor.append(make_1hot(c, token2pos, vocab_size))
    target_tensor.append(make_1hot(_EOS, token2pos, vocab_size))
    return torch.concat(target_tensor).unsqueeze(dim=0)


def make_padded_batch(tensors, lens, vocab_size, max_len):
	"""Given a batch of sequences represented as tensors and a list containing
	their lengths, pads the sequences with zeros and builds a batch."""
	padding_tensor = torch.zeros(vocab_size)
	padding_tensor[-1] = 1
	padded_tensors = [torch.concat((t, torch.tile(padding_tensor, dims=(max_len-l, 1)).unsqueeze(0)), dim=1)
					  for t, l in zip(tensors, lens)]
	return torch.concat(padded_tensors)  # make batch


def reduce_lens(lens):
	"""Given a list of lengths of sequences in a batch, reduce each length
	up to zero to keep track of valid and padded positions."""
	return [l-1 if l > 0 else l for l in lens]


def get_mask(lens, device='cpu'):
	"""Given a list of lengths of sequences, create a tensor mask that signals
	the sequences that reached padded positions."""
	return torch.tensor([1 if l > 0 else 0 for l in lens], device=device)


def get_hidden_mask(lens, h_size, device):
	"""Given a mask for a position in a batch of sequences, produce a mask for
	the batch of hidden states of a recurrent network to be used to process the
	elements in the batch in the current position."""
	mask_1d = get_mask(lens, device)
	return torch.concat(
		[torch.zeros((1, h_size), device=device) if m == 0 else torch.ones((1, h_size), device=device) for m in mask_1d],
		dim=0)

def get_reading_mask(mask_1d, mem_size, device):
	return torch.concat(
		[torch.zeros((1, mem_size), device=device) if m == 0 else torch.ones((1, mem_size), device=device) for m in mask_1d],
		dim=0)


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


def populate_first_output(output, samples_len, first_output):
	for pos, seq_len in enumerate(samples_len):
		if seq_len == 0:
			first_output.setdefault(pos, output[pos, :].unsqueeze(dim=0))
	return first_output


def build_first_output(first_output):
	return torch.concat([output for output in first_output.values()], dim=0)
