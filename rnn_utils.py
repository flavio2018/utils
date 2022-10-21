"""Utilities to train recurrent neural networks on variable-length sequences with PyTorch"""


import torch


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


def make_padded_batch(tensors, lens, vocab_size):
	"""Given a batch of sequences represented as tensors and a list containing
	their lengths, pads the sequences with zeros and builds a batch."""
	padded_tensors = [torch.concat((t, torch.zeros((1, max(lens)-l, vocab_size))), dim=1)
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
		h_dict.setdefault(state.item(), model.h_t_1[state, :])
		c_dict.setdefault(state.item(), model.c_t_1[state, :])
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


def batch_acc(outputs, target, loss_masks):
    masked_outputs = torch.concat(
        [(output.argmax(1)*mask).unsqueeze(1) for output, mask in zip(outputs, loss_masks)], dim=1)
    not_valid_outputs = (masked_outputs == 0).sum()
    valid_outputs = (masked_outputs != 0).sum()
    outputs_equal_to_targets = (masked_outputs == target.argmax(2)).sum()
    return (outputs_equal_to_targets - not_valid_outputs)/valid_outputs


def get_token2pos(vocab_chars):
	token2pos = {t: p for p, t in enumerate(vocab_chars)}
	token2pos['\n'] = len(token2pos)
	return token2pos


def get_pos2token(vocab_chars):
	token2pos = get_token2pos(vocab_chars)
	return {p: t for t, p in token2pos.items()}


def target_tensors_to_str(y_t):
	pos2token = get_pos2token()
	idx_outputs = [torch.argmax(o).item() for o in y_t]
	return ''.join([pos2token[idx] for idx in idx_outputs])


def lstm_fwd_padded_batch(model, sample, target, samples_len, targets_len, device):
	model.eval()
	outputs = []
	h_dict, c_dict = {}, {}
	first_output = {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	hid_size = model.h_t_1.size(1)

	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		output = model(sample[:, char_pos, :].squeeze(), hidden_mask)
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(model, h_dict, c_dict, samples_len)
		first_output = populate_first_output(output, samples_len, first_output)
	outputs.append(build_first_output(first_output))
	
	model.set_states(h_dict, c_dict)
	
	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		output = model(target[:, char_pos, :].squeeze(), hidden_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)
	return outputs


def eval_padded(outputs, target, sample):
	BS = target.size(0)

	samples_str = [target_tensors_to_str([sample[BATCH, p, :] for p in range(sample.size(1))]) for BATCH in range(BS)]	
	targets_str = [target_tensors_to_str([target[BATCH, p, :] for p in range(target.size(1))]) for BATCH in range(BS)]
	outputs_str = [target_tensors_to_str([o[BATCH, :] for o in outputs]) for BATCH in range(BS)]
	
	idx = torch.randint(BS, (1,)).item()
	print(samples_str[idx])
	print("out:", outputs_str[idx])
	print("target:", targets_str[idx])
	print()


def eval_lstm_padded(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device):
	outputs = lstm_fwd_padded_batch(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	eval_padded(outputs, padded_targets_batch, padded_samples_batch)
