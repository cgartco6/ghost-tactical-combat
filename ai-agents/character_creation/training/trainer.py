import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List
import numpy as np
from ...models.character_model import CharacterModel
from ...models.training_data import CharacterDataset, create_training_example_from_character

class CharacterTrainingDataset(Dataset):
    def __init__(self, examples: List, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize description
        encoding = self.tokenizer(
            example.description,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert traits to multi-hot encoding
        all_traits = ['confident', 'aggressive', 'calm', 'intelligent', 'loyal', 
                     'cautious', 'reckless', 'stoic', 'charismatic', 'reserved']
        traits = [1 if trait in example.traits else 0 for trait in all_traits]
        
        # Normalize skills to 0-1 range
        skill_names = ['marksmanship', 'stealth', 'tactics', 'endurance', 
                      'technical', 'medical', 'explosives', 'leadership']
        skills = [example.skills.get(skill, 50) / 100 for skill in skill_names]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'traits': torch.tensor(traits, dtype=torch.float),
            'skills': torch.tensor(skills, dtype=torch.float)
        }

class CharacterTrainer:
    def __init__(self, model: CharacterModel, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.trait_criterion = nn.BCEWithLogitsLoss()
        self.skill_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
    
    def train_epoch(self, dataloader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            traits = batch['traits'].to(self.device)
            skills = batch['skills'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate losses
            lm_loss = self.lm_criterion(
                outputs['lm_logits'].view(-1, outputs['lm_logits'].size(-1)),
                input_ids.view(-1)
            )
            
            trait_loss = self.trait_criterion(
                outputs['trait_logits'],
                traits
            )
            
            skill_loss = self.skill_criterion(
                outputs['skill_values'],
                skills
            )
            
            # Combined loss
            loss = lm_loss + trait_loss + skill_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                traits = batch['traits'].to(self.device)
                skills = batch['skills'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                lm_loss = self.lm_criterion(
                    outputs['lm_logits'].view(-1, outputs['lm_logits'].size(-1)),
                    input_ids.view(-1)
                )
                
                trait_loss = self.trait_criterion(
                    outputs['trait_logits'],
                    traits
                )
                
                skill_loss = self.skill_criterion(
                    outputs['skill_values'],
                    skills
                )
                
                # Combined loss
                loss = lm_loss + trait_loss + skill_loss
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 10):
        """Train the model for multiple epochs"""
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path))

def prepare_training_data(character_data: List[Dict[str, Any]], tokenizer, val_ratio=0.2):
    """Prepare training data from character data"""
    examples = []
    
    for char in character_data:
        example = create_training_example_from_character(char)
        examples.append(example)
    
    # Split into train and validation
    split_idx = int(len(examples) * (1 - val_ratio))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    train_dataset = CharacterTrainingDataset(train_examples, tokenizer)
    val_dataset = CharacterTrainingDataset(val_examples, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_dataloader, val_dataloader
