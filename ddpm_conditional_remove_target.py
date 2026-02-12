import os
import logging
import random
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

from utils_remove_target_source_history import *
from modules import MLP_conditional, TargetAwareAttention, RatingPredictor

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, ddim_steps=100, beta_start=1e-4, beta_end=0.02, lamda=0.9, device="cuda"):
        # Initialize diffusion schedule and core hyperparameters.
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.ddim_steps = ddim_steps
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.num_heads = 4
        self.review_emb_dim = 768
        self.device = device
        self.lamda = lamda

    def prepare_noise_schedule(self):
        # Build linear beta schedule for diffusion steps.
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)


    def noise_embeddings(self, x, t):
        # Add noise to embeddings at timestep t and return noisy x and epsilon.
        alpha_hat_t = self.alpha_hat[t]
        sqrt_alpha_hat = torch.sqrt(alpha_hat_t).unsqueeze(1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat_t).unsqueeze(1)
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    
    def sample_timesteps(self, n):
        # Sample random diffusion timesteps for a batch.
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    

    @torch.inference_mode()
    def sample(self, target_aware_pref):
        # DDIM sampling conditioned on target-aware preference.
        model = self.model
        device = self.device
        n = target_aware_pref.shape[0]

        model.eval()
        self.target_aware_attn.eval()
        self.rating_predictor.eval()

        ddim_steps = self.ddim_steps
        timesteps = torch.linspace(self.noise_steps - 1, 0, steps=ddim_steps, dtype=torch.long).to(device)

        with torch.inference_mode():
            x = torch.randn((n, self.review_emb_dim), device=device)

            for idx in range(len(timesteps) - 1):
                t_cur = int(timesteps[idx].item())
                t_next = int(timesteps[idx + 1].item())

                t = torch.full((n,), t_cur, dtype=torch.long, device=device)

                pred_x0 = model(x, t, target_aware_pref)

                alpha_hat_t = self.alpha_hat[t].unsqueeze(1)
                t_next_tensor = torch.full((n,), t_next, dtype=torch.long, device=device)
                alpha_hat_t_next = self.alpha_hat[t_next_tensor].unsqueeze(1)

                sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
                sqrt_alpha_hat_t_next = torch.sqrt(alpha_hat_t_next)

                sqrt_1m_alpha_hat_t = torch.sqrt(1.0 - alpha_hat_t)
                sqrt_1m_alpha_hat_t_next = torch.sqrt(1.0 - alpha_hat_t_next)

                coef_x0 = sqrt_alpha_hat_t_next - (sqrt_1m_alpha_hat_t_next * sqrt_alpha_hat_t) / (sqrt_1m_alpha_hat_t + 1e-8)
                coef_xt = sqrt_1m_alpha_hat_t_next / (sqrt_1m_alpha_hat_t + 1e-8)

                x = coef_x0 * pred_x0 + coef_xt * x

            return x

    def train_step(self, loss):
        # Run one optimizer step with mixed precision scaling.
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def one_epoch(self, epoch, train=True):
        # Run one training epoch over the dataloader.
        avg_loss = 0.
        loss_sum = 0.
        num_batches = 0
        
        if train: 
            self.model.train()
            self.target_aware_attn.train()
            self.rating_predictor.train()

        else: 
            self.model.eval()
            self.target_aware_attn.eval()
            self.rating_predictor.eval()

        if train:
            pbar = progress_bar(self.train_dataloader, leave=False)
        else:
            pbar = progress_bar(self.val_dataloader, leave=False)
        

        for i, batch in enumerate(pbar):
            with torch.autocast("cuda"), (torch.inference_mode() if not train else torch.enable_grad()):
                target_review = batch['target_review'].to(self.device)
                target_item_embedding = batch['target_item_embedding'].to(self.device)
                target_rating = batch['target_rating'].to(self.device)
                num_target_item_interact = batch['num_target_item_interact'].to(self.device)

                num = num_target_item_interact.unsqueeze(1)
                target_item_embedding = (target_item_embedding - target_review / num) * (num / (num - 1))

                u_i_pair = batch['pair']
                item_ids = torch.tensor([pair[1] for pair in u_i_pair], dtype=torch.long, device=self.device)
                
                source_reviews = [review.to(self.device) for review in batch['source_reviews']]
                source_item_embeddings = [emb.to(self.device) for emb in batch['source_item_embeddings']]

                t = self.sample_timesteps(target_review.shape[0]).to(self.device)
                x_t, noise = self.noise_embeddings(target_review, t)
                
                target_aware_pref, attn_weights = self.target_aware_attn(target_item_embedding, source_item_embeddings, source_reviews)
                if np.random.random() < self.unconditional_prob:
                    target_aware_pref = None
                
                predicted_emb = self.model(x_t, t, target_aware_pref)
                predicted_rating = self.rating_predictor(predicted_emb, item_ids)
                rec_loss = self.mse(target_rating, predicted_rating)
                diffusion_loss = self.mse(target_review, predicted_emb)

                loss = diffusion_loss * self.lamda + rec_loss * (1 - self.lamda)
                avg_loss += loss

                loss_sum += loss.item()
                num_batches += 1
                
            if train:
                self.train_step(loss)

            pbar.comment = f"MSE={loss.item():2.3f}"        



        phase = "train" if train else "val"
        logging.info(
            "%s epoch=%d loss=%.6f",
            phase,
            epoch,
            loss_sum / num_batches,
        )
        return 

    @torch.inference_mode()
    def test(self, val = False):
        # Evaluate MAE/RMSE on validation or test split.
        self.model.eval()
        total_mae  = 0.0
        total_rmse = 0.0       
        total_users = 0


        if val:
            pbar = progress_bar(self.val_dataloader, leave=False)
        else:
            pbar = progress_bar(self.test_dataloader, leave=False)
        
        for batch in pbar:
            target_item_embedding = batch['target_item_embedding'].to(self.device)
            target_rating = batch['target_rating'].to(self.device)
            u_i_pair = batch['pair']
            item_ids = torch.tensor([pair[1] for pair in u_i_pair], dtype=torch.long, device=self.device)
            source_reviews = [review.to(self.device) for review in batch['source_reviews']]
            source_item_embeddings = [emb.to(self.device) for emb in batch['source_item_embeddings']]


            target_aware_pref, attn_weights = self.target_aware_attn(target_item_embedding, source_item_embeddings, source_reviews)
            predicted_emb = self.sample(target_aware_pref=target_aware_pref)
            predicted_rating = self.rating_predictor(predicted_emb, item_ids)
            
            predicted_rating = torch.clamp(predicted_rating, min=1, max=5)

            batch_ae, batch_mse = compute_rec(predicted_rating, target_rating)

            
            batch_size = predicted_rating.shape[0]
           
            total_mae   += batch_ae
            total_rmse  += batch_mse

            total_users += batch_size

        avg_mae  = total_mae  / total_users
        avg_rmse = (total_rmse / total_users) ** 0.5



        return avg_mae, avg_rmse

    def save_model(self, run_name, epoch=-1):
        # Save model, optimizer, and RNG states to checkpoint.
        save_dir = os.path.join("models", run_name)
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'target_aware_attn': self.target_aware_attn.state_dict(),
            'rating_predictor': self.rating_predictor.state_dict(),
            'epoch': epoch,
            'rng_state': torch.get_rng_state().cpu().numpy().tolist(),
            'cuda_rng_state': [state.cpu().numpy().tolist() for state in torch.cuda.get_rng_state_all()],
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }

        torch.save(checkpoint, os.path.join(save_dir, "ckpt.pt"))


    def load_model(self, model_cpkt_path, model_ckpt=None):
        # Restore model, optimizer, and RNG states from checkpoint.
        if model_ckpt is None:
            model_ckpt = "ckpt.pt"
        checkpoint = torch.load(os.path.join(model_cpkt_path, model_ckpt), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.target_aware_attn.load_state_dict(checkpoint['target_aware_attn'])
        self.rating_predictor.load_state_dict(checkpoint['rating_predictor'])


        if 'rng_state' in checkpoint:
            torch.set_rng_state(torch.tensor(checkpoint['rng_state'], dtype=torch.uint8))
        if 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state_all([
                torch.tensor(state, dtype=torch.uint8) for state in checkpoint['cuda_rng_state']
            ])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])

    def prepare(self, args):
        # Build dataloaders, models, and optimizers; load pretraining if enabled.
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader, self.test_dataloader, n_user, n_item = get_data(args)
        self.unconditional_prob = getattr(args, 'unconditional_prob', 0.1)

        self.target_aware_attn = TargetAwareAttention(self.review_emb_dim, self.num_heads).to(self.device)
        self.model = MLP_conditional(self.review_emb_dim, condition=True).to(self.device)
        self.rating_predictor = RatingPredictor(self.review_emb_dim, n_user, n_item).to(self.device)

        if getattr(args, 'pretraining', False):
            ckpt_path = os.path.join("models", "pretraining" + "_" + args.dataset + str(args.cold_start_ratio), "ckpt.pt")
            if not os.path.exists(ckpt_path):
                logging.warning(f"Pretraining checkpoint not found at {ckpt_path}")
            else:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model_state = checkpoint.get('model')
                rating_state = checkpoint.get('rating_predictor')
                if model_state is not None:
                    self.model.load_state_dict(model_state, strict=False)
                if rating_state is not None:
                    self.rating_predictor.load_state_dict(rating_state, strict=False)
                logging.info(f"Loaded pretraining weights for model and rating_predictor from {ckpt_path}")
        else:
            logging.info(f"No pretraining weights")
        
        all_params = list(self.model.parameters()) + list(self.target_aware_attn.parameters()) + list(self.rating_predictor.parameters())
        
        self.optimizer = optim.AdamW(all_params, lr=args.lr, eps=1e-5, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.scaler = torch.amp.GradScaler()


    def fit(self, args):
        # Train with early stopping and final evaluation on test set.
        best_val_score = float('inf')
        patience_counter = 0
        early_stopping_patience = args.early_stop

        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            self.one_epoch(epoch=epoch, train=True)
            if args.do_validation:
                val_mae, val_rmse = self.test(val=True)
                logging.info(
                    "val epoch=%d MAE=%.6f RMSE=%.6f",
                    epoch,
                    val_mae,
                    val_rmse,
                )

                val_score = 0.5 * val_rmse + 0.5 * val_mae
                if val_score < best_val_score:
                    best_val_score = val_score
                    patience_counter = 0

                    self.save_model(run_name=args.run_name, epoch=epoch)
                else:
                    patience_counter += 1
                    logging.info(f"No improvement in validation. Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break
            

        logging.info("Training completed. Loading best model for final test evaluation...")
        self.load_model(model_cpkt_path=os.path.join("models", args.run_name))
        final_test_mae, final_test_rmse = self.test(val=False)

        logging.info(f"Final test results -> MAE: {final_test_mae:.4f}, RMSE: {final_test_rmse:.4f}")
        logging.info(
            "final test MAE=%.6f RMSE=%.6f",
            final_test_mae,
            final_test_rmse,
        )
