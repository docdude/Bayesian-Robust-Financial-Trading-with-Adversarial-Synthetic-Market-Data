# -*- coding: UTF-8 -*-
# Local modules
import os
import pickle
from typing import Dict, Union

# 3rd party modules
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# Self-written modules
from models.dataset import TimeGANDataset

import pandas as pd

def _save_checkpoint(model, args, phase, epoch):
    """Save a training checkpoint."""
    ckpt_dir = os.path.join(args.model_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{phase}_epoch{epoch}.pt")
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")


def _find_latest_checkpoint(ckpt_dir, phase):
    """Find the latest checkpoint for a given phase."""
    if not os.path.exists(ckpt_dir):
        return None, 0
    import re
    best_epoch = 0
    best_path = None
    for f in os.listdir(ckpt_dir):
        m = re.match(rf"{phase}_epoch(\d+)\.pt", f)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = os.path.join(ckpt_dir, f)
    return best_path, best_epoch


def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None,
    start_epoch: int = 0
) -> None:
    """The training loop for the embedding and recovery functions
    """  
    logger = trange(start_epoch, args.emb_epochs, desc=f"Epoch: {start_epoch}, Loss: 0")
    for epoch in logger:   
        for X_mb, T_mb, M_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # print("X_mb",X_mb)

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb,M=M_mb, Z=None, obj="autoencoder")
            if torch.cuda.device_count() > 1:
                E_loss_T0 = E_loss_T0.mean()
                E_loss0 = E_loss0.mean()
            loss = np.sqrt(E_loss_T0.item())

            # Backward Pass
            E_loss0.backward()

            # Update model parameters
            e_opt.step()
            r_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Embedding/Loss:", 
                loss, 
                epoch
            )
            writer.flush()
        if (epoch + 1) % 50 == 0:
            _save_checkpoint(model, args, "emb", epoch + 1)

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None,
    start_epoch: int = 0
) -> None:
    """The training loop for the supervisor function
    """
    logger = trange(start_epoch, args.sup_epochs, desc=f"Epoch: {start_epoch}, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb, M_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, T=T_mb,M=M_mb, Z=None, obj="supervisor")
            if torch.cuda.device_count() > 1:
                S_loss = S_loss.mean()

            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            s_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Supervisor/Loss:", 
                loss, 
                epoch
            )
            writer.flush()
        if (epoch + 1) % 50 == 0:
            _save_checkpoint(model, args, "sup", epoch + 1)

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None,
    start_epoch: int = 0,
) -> None:
    """The training loop for training the model altogether
    """
    logger = trange(
        start_epoch, args.gan_epochs, 
        desc=f"Epoch: {start_epoch}, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    
    for epoch in logger:
        for X_mb, T_mb,M_mb in dataloader:
            ## Generator Training
            for _ in range(2):
                # Random Generator
                if args.hist_mode=="dim1":
                    Z_mb = torch.rand((X_mb.size()[0], args.max_seq_len, args.Z_dim))

                    # Z_mb_ is X_mb with the first args.Z_dim columns replaced with Z_mb
                    Z_mb_ = torch.cat((Z_mb, X_mb[:, :, args.Z_dim:]), dim=2)
                elif args.hist_mode=="dim0":
                    Z_mb = torch.rand((X_mb.size()[0], args.max_seq_len//2, args.Z_dim))
                    # repace the second half of the real data with noise and keep the second half as it 
                    Z_= np.concatenate((X_mb[:,:args.max_seq_len//2,:args.Z_dim],Z_mb),axis=1)
                    # to tensor
                    Z_= torch.tensor(Z_).to(X_mb.device)
                    Z_mb_ = torch.cat((Z_,X_mb[:,:,args.Z_dim:]),dim=2)

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss,G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, KL_loss, STD_loss, MS_loss_masked = model(X=X_mb, T=T_mb,M=M_mb, Z=Z_mb_, obj="generator")
                if torch.cuda.device_count() > 1:
                    G_loss = G_loss.mean()
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb,M=M_mb, Z=Z_mb_, obj="autoencoder")
                if torch.cuda.device_count() > 1:
                    E_loss = E_loss.mean()
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())
                
                # Update model parameters
                e_opt.step()
                r_opt.step()

            # Random Generator
            if args.hist_mode=="dim1":
                Z_mb = torch.rand((X_mb.size()[0], args.max_seq_len, args.Z_dim))

                # Z_mb_ is X_mb with the first args.Z_dim columns replaced with Z_mb
                Z_mb_ = torch.cat((Z_mb, X_mb[:, :, args.Z_dim:]), dim=2)
            elif args.hist_mode=="dim0":
                Z_mb = torch.rand((X_mb.size()[0], args.max_seq_len//2, args.Z_dim))
                # repace the second half of the real data with noise and keep the second half as it 
                Z_= np.concatenate((X_mb[:,:args.max_seq_len//2,:args.Z_dim],Z_mb),axis=1)
                # to tensor
                Z_= torch.tensor(Z_).to(X_mb.device)
                Z_mb_ = torch.cat((Z_,X_mb[:,:,args.Z_dim:]),dim=2)

            ## Discriminator Training
            model.zero_grad()
            # Forward Pass
            D_loss = model(X=X_mb, T=T_mb,M=M_mb, Z=Z_mb_, obj="discriminator")
            if torch.cuda.device_count() > 1:
                D_loss = D_loss.mean()

            # Check Discriminator loss
            if D_loss > args.dis_thresh:
                # Backward Pass
                D_loss.backward()

                # Update model parameters
                d_opt.step()
            D_loss = D_loss.item()

        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f},G_loss_U: {G_loss_U:.4f}, G_loss_U_e: {G_loss_U_e:.4f}, G_loss_S: {G_loss_S:.4f}, G_loss_V: {G_loss_V:.4f}, KL_loss: {KL_loss:.4f}, STD_loss: {STD_loss:.4f}, MS_loss_masked: {MS_loss_masked:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Joint/Embedding_Loss:', 
                E_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Generator_Loss:', 
                G_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Discriminator_Loss:', 
                D_loss, 
                epoch
            )
            writer.flush()
        if (epoch + 1) % 50 == 0:
            _save_checkpoint(model, args, "gan", epoch + 1)

def timegan_trainer(model, data, time,mask, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    # Initialize TimeGAN dataset and dataloader
    dataset = TimeGANDataset(data, time, mask)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Set to False or True depending on your needs
        num_workers=torch.cuda.device_count(),  # Optional: Set to the number of CPU cores for data loading
        pin_memory=True  # Optional: Set to True to speed up the transfer of data to the GPU
    )

    # model.to(args.device)
    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(args.device)  # Send the model to the GPU(s)

    # # Initialize Optimizers
    # e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    # r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    # s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    # g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    # d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

    # Initialize Optimizers after DataParallel wrapping
    if torch.cuda.device_count() > 1:
        e_opt = torch.optim.Adam(model.module.embedder.parameters(), lr=args.learning_rate)
        r_opt = torch.optim.Adam(model.module.recovery.parameters(), lr=args.learning_rate)
        s_opt = torch.optim.Adam(model.module.supervisor.parameters(), lr=args.learning_rate)
        g_opt = torch.optim.Adam(model.module.generator.parameters(), lr=args.learning_rate)
        d_opt = torch.optim.Adam(model.module.discriminator.parameters(), lr=args.learning_rate)
    else:
        e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
        r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
        s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
        g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
        d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    # --- Resume logic: detect existing checkpoints ---
    ckpt_dir = os.path.join(args.model_path, "checkpoints")
    resume = getattr(args, 'resume', False)

    emb_ckpt_path, emb_ckpt_epoch = (None, 0)
    sup_ckpt_path, sup_ckpt_epoch = (None, 0)
    gan_ckpt_path, gan_ckpt_epoch = (None, 0)
    if resume:
        emb_ckpt_path, emb_ckpt_epoch = _find_latest_checkpoint(ckpt_dir, "emb")
        sup_ckpt_path, sup_ckpt_epoch = _find_latest_checkpoint(ckpt_dir, "sup")
        gan_ckpt_path, gan_ckpt_epoch = _find_latest_checkpoint(ckpt_dir, "gan")
        # Load the most advanced checkpoint into the model
        load_path = gan_ckpt_path or sup_ckpt_path or emb_ckpt_path
        if load_path:
            print(f"\nResuming from checkpoint: {load_path}")
            state = torch.load(load_path, map_location=args.device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state)
            else:
                model.load_state_dict(state)

    # Phase 1: Embedding
    if resume and emb_ckpt_epoch >= args.emb_epochs:
        print(f"\nSkipping Embedding Training (checkpoint at epoch {emb_ckpt_epoch})")
    else:
        emb_start = emb_ckpt_epoch if resume and emb_ckpt_epoch > 0 else 0
        print(f"\nStart Embedding Network Training (from epoch {emb_start})")
        embedding_trainer(
            model=model, 
            dataloader=dataloader, 
            e_opt=e_opt, 
            r_opt=r_opt, 
            args=args, 
            writer=writer,
            start_epoch=emb_start
        )

    # Phase 2: Supervisor
    if resume and sup_ckpt_epoch >= args.sup_epochs:
        print(f"\nSkipping Supervisor Training (checkpoint at epoch {sup_ckpt_epoch})")
    else:
        sup_start = sup_ckpt_epoch if resume and sup_ckpt_epoch > 0 else 0
        print(f"\nStart Training with Supervised Loss Only (from epoch {sup_start})")
        supervisor_trainer(
            model=model,
            dataloader=dataloader,
            s_opt=s_opt,
            g_opt=g_opt,
            args=args,
            writer=writer,
            start_epoch=sup_start
        )

    # Phase 3: Joint (adversarial)
    if resume and gan_ckpt_epoch >= args.gan_epochs:
        print(f"\nSkipping Joint Training (checkpoint at epoch {gan_ckpt_epoch})")
    else:
        gan_start = gan_ckpt_epoch if resume and gan_ckpt_epoch > 0 else 0
        print(f"\nStart Joint Training (from epoch {gan_start})")
        joint_trainer(
            model=model,
            dataloader=dataloader,
            e_opt=e_opt,
            r_opt=r_opt,
            s_opt=s_opt,
            g_opt=g_opt,
            d_opt=d_opt,
            args=args,
            writer=writer,
            start_epoch=gan_start,
        )

    # Save model, args, and hyperparameters
    # torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}")

def timegan_generator(model, T, args,history,macro,real_data):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        args = pickle.load(fb)

    # model = nn.DataParallel(model)

    # model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))

    # Load the state_dict
    state_dict = torch.load(f"{args.model_path}/model.pt")

    # Remove "module." prefix from the state_dict keys
    if torch.cuda.device_count() > 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove "module." from the key
            else:
                new_state_dict[k] = v

        # Load the modified state_dict into the model
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    print("\nGenerating Data...")
    # Initialize model to evaluation mode and run without gradients
    # model.to(args.device)
    # Wrap the model with DataParallel if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    model.to(args.device)  # Send the model to the GPU(s)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        if args.hist_mode=="dim1":
            noise = torch.rand((len(T), args.max_seq_len, args.Z_dim))
            # cast macro and history to tensor
            # print size of noise, macro and history
            Z = np.concatenate((noise,history,macro),axis=2)
        elif args.hist_mode=="dim0":
            noise = torch.rand((len(T), args.max_seq_len//2, args.Z_dim))
            # repace the second half of the real data with noise and keep the second half as it 
            Z_= np.concatenate((real_data[:,:args.max_seq_len//2,:args.Z_dim],noise),axis=1)
            Z = np.concatenate((Z_,real_data[:,:,args.Z_dim:]),axis=2)
        # add noise to the first args.Z_dim columns of Z
        # noise = torch.rand((len(T), args.max_seq_len, args.Z_dim))
        # Z[:, :, :args.Z_dim] = noise + Z[:, :, :args.Z_dim]

        #for test
        # Z = torch.rand((len(T), args.max_seq_len, args.input_data_dim))
        # print("T",T)
        # print("T type",type(T))
        # print("T shape",T.shape)

        # cast T to tensor
        T = torch.tensor(T).to(args.device)
        
        generated_data = model(X=None, T=T, Z=Z,M=None,obj="inference")

    return generated_data.numpy()


