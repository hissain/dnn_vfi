import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    img_size: Tuple[int, int] = (128, 128)
    num_interpolated: int = 1
    batch_size: int = 8
    epochs: int = 10
    hidden_dim: int = 256
    num_layers: int = 4
    learning_rate: float = 1e-4
    train_ratio: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "output/model"

    @property
    def input_dim(self) -> int:
        return self.img_size[0] * self.img_size[1] * 3

class VideoDataset(Dataset):
    def __init__(self, frames: torch.Tensor, sequence_length: int):
        self.frames = frames
        self.sequence_length = sequence_length
        self.num_sequences = len(frames) - sequence_length + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx:idx + self.sequence_length]

class S6Layer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual

class VFIMamba(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, config.hidden_dim, kernel_size=3, padding=1)
        )

        reduced_dim = config.hidden_dim // 2
        self.s6_layers = nn.ModuleList([
            S6Layer(reduced_dim) for _ in range(config.num_layers)
        ])

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Conv2d(config.hidden_dim, reduced_dim, 1)
        self.dim_expand = nn.Conv2d(reduced_dim, config.hidden_dim, 1)

    def process_sequence(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.s6_layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv_encoder(x)

        _, C_hidden, H_hidden, W_hidden = x.shape
        x = self.dim_reduce(x)
        x = x.view(B, T, -1, H_hidden, W_hidden)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * H_hidden * W_hidden, T, -1)
        x = self.process_sequence(x)
        x = x.view(B, H_hidden, W_hidden, T, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * T, -1, H_hidden, W_hidden)
        x = self.dim_expand(x)
        x = self.conv_decoder(x)
        x = x.view(B, T, C, H, W)
        return x

class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(self.config.device)
            
            start_frames = batch[:, [0]]
            end_frames = batch[:, [-1]]
            target_frames = batch[:, 1:-1]
            input_frames = torch.cat([start_frames, end_frames], dim=1)
            
            pred_frames = self.model(input_frames)
            loss = self.criterion(pred_frames, target_frames)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(dataloader)

    def create_visualization(self, input_frames: torch.Tensor, predicted_frames: torch.Tensor) -> HTML:
        plt.ioff()
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.close()

        # Convert tensors from (B, T, C, H, W) to (B, T, H, W, C)
        def prepare_frames(frames):
            frames = frames.cpu().detach().numpy()
            frames = np.transpose(frames, (0, 1, 3, 4, 2))
            return frames

        # Prepare frames for the first 9 sequences
        input_frames = prepare_frames(input_frames[:9])  # Take first 9 samples
        predicted_frames = prepare_frames(predicted_frames[:9])

        # Create sequence of 3 frames for each sample: frame1 -> predicted -> frame2
        def create_sequence(idx):
            return [
                (input_frames[idx, 0], "Frame 1"),
                (predicted_frames[idx, 0], "Predicted"),
                (input_frames[idx, 1], "Frame 2")
            ]

        sequences = [create_sequence(i) for i in range(min(9, len(input_frames)))]
        frame_idx = 0

        def animate(t):
            nonlocal frame_idx
            frame_idx = (frame_idx + 1) % 3  # Cycle through 3 frames

            for i in range(3):
                for j in range(3):
                    seq_idx = i * 3 + j
                    if seq_idx < len(sequences):
                        ax = axes[i, j]
                        ax.clear()
                        ax.axis('off')
                        
                        # Get current frame from sequence
                        frame, title = sequences[seq_idx][frame_idx]
                        frame = np.clip(frame, 0, 1)
                        ax.imshow(frame)
                        ax.set_title(f'Sequence {seq_idx + 1}: {title}')

            plt.tight_layout()

        # Create animation with faster playback
        anim = FuncAnimation(
            fig, animate, frames=30,  # More frames for smoother loop
            interval=300,  # 300ms between frames
            repeat=True,
            blit=False
        )
        
        plt.close()
        return HTML(anim.to_jshtml(default_mode='loop'))

    def save_model(self) -> None:
        """Save the trained model and optimizer state."""
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, save_path / 'vfi_model.pth')
        logger.info(f"Model saved to {save_path / 'vfi_model.pth'}")

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, HTML]:
        self.model.eval()
        
        # Get 9 test samples
        test_samples = []
        for batch in test_loader:
            test_samples.append(batch)
            if len(test_samples) * batch.size(0) >= 9:
                break
                
        # Concatenate batches and take first 9 samples
        test_batch = torch.cat(test_samples, dim=0)[:9]
        
        with torch.no_grad():
            test_batch = test_batch.to(self.config.device)
            start_frames = test_batch[:, [0]]
            end_frames = test_batch[:, [-1]]
            target_frames = test_batch[:, 1:-1]
            input_frames = torch.cat([start_frames, end_frames], dim=1)
            
            pred_frames = self.model(input_frames)
            loss = self.criterion(pred_frames, target_frames)
            
            visualization = self.create_visualization(input_frames, pred_frames)
            
        return loss.item(), visualization

def load_video(video_path: str, config: Config) -> List[np.ndarray]:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Loading video frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, config.img_size)
            frames.append(frame)
    finally:
        cap.release()
        
    return frames

def prepare_data(frames: List[np.ndarray], config: Config) -> Tuple[DataLoader, DataLoader]:
    frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    
    sequence_length = config.num_interpolated + 2
    dataset = VideoDataset(frames_tensor, sequence_length)
    
    train_size = int(len(dataset) * config.train_ratio)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0 if config.device == "cpu" else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0 if config.device == "cpu" else 4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def run_training(video_path: str, config: Optional[Config] = None) -> HTML:
    if config is None:
        config = Config()
    
    logger.info(f"Using device: {config.device}")
    logger.info("Loading video and preparing data...")
    
    frames = load_video(video_path, config)
    train_loader, test_loader = prepare_data(frames, config)
    
    logger.info("Initializing model and trainer...")
    model = VFIMamba(config)
    trainer = Trainer(model, config)
    
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - Loss: {loss:.4f}")
    
    logger.info("Evaluating model and generating visualization...")
    test_loss, visualization = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    trainer.save_model()
    
    return visualization