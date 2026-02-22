"""
Training utilities for AneryzAI LLM
Includes dataset handling, loss computation, and training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

class LanguageModelDataset(Dataset):
    """Dataset for language modeling with support for multiple languages"""
    
    def __init__(
        self,
        token_ids_list: List[List[int]],
        max_length: int = 2048,
        stride: int = None,
    ):
        """
        Args:
            token_ids_list: List of tokenized documents
            max_length: Maximum sequence length
            stride: Sliding window stride (if None, no overlap)
        """
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.examples = []
        
        # Create training examples from documents
        for tokens in token_ids_list:
            # Use sliding window to create multiple examples from one document
            for i in range(0, len(tokens) - max_length, self.stride):
                self.examples.append(tokens[i : i + max_length + 1])
            
            # Add last segment if it's long enough
            if len(tokens) >= max_length:
                self.examples.append(tokens[-max_length - 1 :])
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with input_ids and labels
        """
        tokens = self.examples[idx]
        
        # Input is all but last token, target is all but first token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad to max_length
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.fill(torch.empty(padding_length, dtype=torch.long), -100)  # Ignore padding in loss
            ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss untuk mencegah overfitting
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len) dengan value -100 untuk tokens yang diabaikan
        """
        # Reshape for loss computation
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # Filter out ignored indices
        valid_idx = targets != -100
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Create smoothed target distribution
        one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
        smoothed_target = (
            one_hot * self.confidence +
            (1 - self.confidence) / (self.vocab_size - 1) * (1 - one_hot)
        )
        
        # Compute KL divergence loss
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = self.kl_div(log_probs, smoothed_target)
        
        return loss


class Trainer:
    """Training orchestrator for the LLM"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        scheduler,
        device: str = "cuda",
        config = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=model.config.vocab_size,
            smoothing=0.1
        ).to(device)
        
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            disable=False
        )
        
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            output = self.model(input_ids)
            logits = output["logits"]
            
            # Compute loss (scale by accumulation steps for proper averaging)
            loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            accumulation_steps += 1
            
            # Only update weights after accumulation steps
            if accumulation_steps == self.gradient_accumulation_steps:
                # Gradient clipping
                if self.config and self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                accumulation_steps = 0
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                avg_loss = total_loss / max(1, num_batches)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Log periodically
                if self.config and self.global_step % self.config.log_steps == 0:
                    print(f"\nStep {self.global_step}, Loss: {avg_loss:.4f}")
                
                # Validation step
                if (
                    self.val_dataloader and
                    self.config and
                    self.global_step % self.config.eval_steps == 0
                ):
                    val_loss = self.validate()
                    print(f"Validation Loss: {val_loss:.4f}")
            else:
                # Accumulate loss for display
                total_loss += loss.item() * self.gradient_accumulation_steps
        
        # Handle remaining accumulated gradients
        if accumulation_steps > 0:
            if self.config and self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            num_batches += 1
            self.global_step += 1
        
        avg_epoch_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_epoch_loss)
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        return avg_epoch_loss
    
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=True):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                output = self.model(input_ids)
                logits = output["logits"]
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        self.model.train()
        return avg_loss
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """
        Main training loop
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model
        """
        best_val_loss = float('inf')
        
        try:
            for epoch in range(num_epochs):
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*50}")
                
                train_loss = self.train_epoch()
                print(f"Train Loss: {train_loss:.4f}")
                
                if self.val_dataloader:
                    val_loss = self.validate()
                    print(f"Val Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if save_path and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Saved best model with val_loss: {val_loss:.4f}")
                        
        except KeyboardInterrupt:
            print(f"\n{'='*50}")
            print("Training interrupted by user!")
            print("Saving current model state...")
            print(f"{'='*50}")
            
            # Save interrupted model
            interrupt_path = "checkpoints/interrupted_model.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': epoch if 'epoch' in locals() else 0,
                'global_step': self.global_step,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, interrupt_path)
            print(f"Model saved to {interrupt_path}")
            print(f"Epoch: {epoch if 'epoch' in locals() else 0}, Global step: {self.global_step}")
            
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"{'='*50}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
) -> optim.Optimizer:
    """Create optimizer with weight decay"""
    
    # Separate parameters with and without weight decay
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int = 10000,
    max_steps: int = 100000,
):
    """Create learning rate scheduler with warmup"""
    
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine annealing
        return max(
            0.0,
            0.5 * (
                1.0 + np.cos(
                    np.pi * float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                )
            ),
        )
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
