import logging
import wandb
import torch


def log_weights_gradient(model, step):
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            # wandb.log({f"hist_gradient/{param_name}": wandb.Histogram(param.grad.cpu())})
            norm = torch.norm(param.grad.detach(), 2)
            wandb.log({
                f"norm2_gradient/{param_name}": norm,
                "step": step,
                })
        else:
            logging.warning(f"{param_name} gradient is None!")
    # code adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    norm_type=2
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    wandb.log({
        f"norm2_gradient/global": total_norm,
        "step": step,
        })

def log_preds_and_targets(batch_i, output, targets):
    if batch_i == 0:
        columns = ["Predictions", "Targets"]
        data = zip([str(p.item()) for p in output.argmax(axis=0)],
                   [str(t.item()) for t in targets])
        data = [list(row) for row in data]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"First batch preds vs targets": table})


def log_config(cfg_dict):
    for subconfig_name, subconfig_values in cfg_dict.items():
        if isinstance(subconfig_values, dict):
            wandb.config.update(subconfig_values)
        else:
            logging.warning(f"{subconfig_name} is not being logged.")


def log_mem_stats(model, step):
    wandb.log({
            "memory_reading/min": model.memory_reading.min(),
            "memory_reading/max": model.memory_reading.max(),
            "memory_reading/2norm": torch.norm(model.memory_reading, 2),
            "step": step,
        })

def log_params_norm(model, step):
    for param_name, param in model.named_parameters():
        wandb.log({
                f"params_norm/{param_name}": torch.norm(param),
                "update": step,
            })


def log_buffers_norm(model, step):
    for buffer_name, buffer in model.named_buffers():
        wandb.log({
                f"buffers_norm/{buffer_name}": torch.norm(buffer),
                "update": step
            })


def log_intermediate_values_norm(model, step):
        wandb.log({
                # read head
                "q_r": torch.norm(model.memory.read_head.query),
                "beta_r": torch.norm(model.memory.read_head.sharpening_beta),
                "w_tilde_r": torch.norm(model.memory.read_head.similarity_vector),
                "gamma_r": torch.norm(model.memory.read_head.lru_gamma),
                "w_hat_r": torch.norm(model.memory.read_head.lru_similarity_vector),
                "v_r": torch.norm(model.memory.read_head.exp_mov_avg_similarity),
                # write head
                "q_w": torch.norm(model.memory.write_head.query),
                "beta_w": torch.norm(model.memory.write_head.sharpening_beta),
                "w_tilde_w": torch.norm(model.memory.write_head.similarity_vector),
                "gamma_w": torch.norm(model.memory.write_head.lru_gamma),
                "w_hat_w": torch.norm(model.memory.write_head.lru_similarity_vector),
                "v_w": torch.norm(model.memory.write_head.exp_mov_avg_similarity),
                # memory
                "w_r": torch.norm(model.memory.read_weights),
                "w_w": torch.norm(model.memory.write_weights),
                "e": torch.norm(model.memory.erase_vector),
                "alpha": torch.norm(model.memory.alpha),
                "c": torch.norm(model.memory.candidate_content_vector),
                "M_c": torch.norm(model.memory.memory_contents),
                "update": step
            })
