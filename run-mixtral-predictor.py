# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/30
import os
import random

import torch
import wandb
from fire import Fire
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm


def train_mixtral_ffn_cosine_similarity_predictor(
        ffn_block_id: int,
        data_dir: str = "/data/data7/pingzhi/data/ffn_input_output_pairs",
        data_with_residual: bool = True,
        save_dir: str = "/data/data8/pingzhi/data/checkpoints",
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        hidden_dim: int = 1024,
        val_ratio: float = 0.1,
        early_stop: int = 5,
):
    wandb.init(
        project="mixtral-ffn-cosine-predictor",
        name=f"ffn-residual-block-{ffn_block_id}" if data_with_residual else f"ffn-block-{ffn_block_id}",
    )

    predictor = nn.Sequential(
        nn.Linear(4096, hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1, bias=False),
        nn.Tanh(),
    )
    predictor = predictor.bfloat16().cuda()
    optimizer = AdamW(predictor.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.MSELoss()

    if data_with_residual:
        data = torch.load(os.path.join(data_dir, f"model.layers.{ffn_block_id}.pt"))
    else:
        data = torch.load(os.path.join(data_dir, f"model.layers.{ffn_block_id}.block_sparse_moe.pt"))
    save_dir = os.path.join(save_dir, f"ffn_residual_block_{ffn_block_id}") if data_with_residual else os.path.join(
        save_dir, f"ffn_block_{ffn_block_id}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Warning: {save_dir} already exists")

    # random split
    random.shuffle(data)
    val_size = int(val_ratio * len(data))
    train_data = data[val_size:]
    val_data = data[:val_size]

    best_val_loss = float("inf")
    early_stop_counter = 0

    progress_bar = tqdm(range(num_epochs * len(train_data)), desc="Training cosine similarity predictors...")

    for epoch in range(num_epochs):
        for batch in train_data:
            optimizer.zero_grad()
            ffn_input, ffn_output = batch
            ffn_input = ffn_input.squeeze().cuda()
            ffn_output = ffn_output.squeeze().cuda()
            with torch.no_grad():
                cos_sim_gt = F.cosine_similarity(ffn_input, ffn_output, dim=-1)

            cos_sim_pred = predictor(ffn_input).squeeze()
            loss = criterion(cos_sim_pred, cos_sim_gt)
            loss.backward()
            optimizer.step()
            progress_bar.update()
            wandb.log({"train_loss": loss.item()})

        val_loss = 0
        for batch in val_data:
            ffn_input, ffn_output = batch
            ffn_input = ffn_input.squeeze().cuda()
            ffn_output = ffn_output.squeeze().cuda()
            with torch.no_grad():
                cos_sim_gt = F.cosine_similarity(ffn_input, ffn_output, dim=-1)
                cos_sim_pred = predictor(ffn_input).squeeze()
            val_loss += criterion(cos_sim_pred, cos_sim_gt).item()

        val_loss /= len(val_data)
        wandb.log({"val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(predictor.state_dict(), os.path.join(save_dir, f"best.pt"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print(f"Early stopped at epoch {epoch}/{num_epochs}")
                break

        # torch.save(predictor.state_dict(), os.path.join(save_dir, f"epoch-{epoch}.pt"))

    torch.save(predictor.state_dict(), os.path.join(save_dir, f"last.pt"))
    wandb.finish()


def eval_mixtral_ffn_cosine_similarity_predictor(
        ffn_block_id: int,
        data_dir: str = "/data/data8/pingzhi/data/ffn_input_output_pairs/testset",
        data_with_residual: bool = False,
        checkpoint_dir: str = "/data/data4/pingzhi/data/checkpoints",
        hidden_dim: int = 1024,
):
    predictor = nn.Sequential(
        nn.Linear(4096, hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1, bias=False),
        nn.Tanh(),
    )
    checkpoint_name = f"ffn_residual_block_{ffn_block_id}" if data_with_residual else f"ffn_block_{ffn_block_id}"
    predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{checkpoint_name}/best.pt")))
    predictor = predictor.bfloat16().cuda()
    predictor.eval()

    if data_with_residual:
        data = torch.load(os.path.join(data_dir, f"model.layers.{ffn_block_id}.pt"))
    else:
        data = torch.load(os.path.join(data_dir, f"model.layers.{ffn_block_id}.block_sparse_moe.pt"))
    cos_sim_pred_list = []

    for batch in tqdm(data, desc="Evaluating cosine similarity predictor..."):
        ffn_input, _ = batch
        ffn_input = ffn_input.squeeze().cuda()
        with torch.no_grad():
            cos_sim_pred = predictor(ffn_input).squeeze()
        cos_sim_pred_list.append(cos_sim_pred)
    cos_sim_pred_list = torch.cat(cos_sim_pred_list)
    average_cos_sim_pred = cos_sim_pred_list.mean().item()

    print(f"[Block {ffn_block_id}] Average predicted output-input cosine similarity: {average_cos_sim_pred}")

    return average_cos_sim_pred


def main_eval():
    cos_sims = []
    for i in range(32):
        avg_sim = eval_mixtral_ffn_cosine_similarity_predictor(ffn_block_id=i)
        cos_sims.append(avg_sim)
    print(cos_sims)


if __name__ == "__main__":
    Fire(main_eval)
