import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from typing import Dict, Any

class CharacterModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 12):
        super(CharacterModel, self).__init__()
        
        # GPT-2 based architecture for character generation
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=12,
            n_positions=1024
        )
        
        self.transformer = GPT2Model(self.config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Personality trait prediction
        self.trait_classifier = nn.Linear(hidden_size, 10)  # 10 personality traits
        
        # Skill prediction
        self.skill_regressor = nn.Linear(hidden_size, 8)  # 8 skills
    
    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Personality trait classification (using the [CLS] token)
        trait_logits = self.trait_classifier(hidden_states[:, 0, :])
        
        # Skill regression (using the [CLS] token)
        skill_values = torch.sigmoid(self.skill_regressor(hidden_states[:, 0, :]))
        
        return {
            'lm_logits': lm_logits,
            'trait_logits': trait_logits,
            'skill_values': skill_values
        }
    
    def generate_character_description(self, input_ids, max_length=100, temperature=1.0):
        """Generate character description using the model"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids
            
            for _ in range(max_length):
                outputs = self.forward(generated)
                next_token_logits = outputs['lm_logits'][:, -1, :] / temperature
                
                # Apply softmax to get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to the generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
            
            return generated
    
    def predict_traits(self, input_ids):
        """Predict personality traits from input"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids)
            trait_probs = torch.softmax(outputs['trait_logits'], dim=-1)
            
            return trait_probs
    
    def predict_skills(self, input_ids):
        """Predict skills from input"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids)
            skills = outputs['skill_values']
            
            return skills

def create_character_model(vocab_size: int, pretrained_path: str = None):
    """Create and optionally load a pretrained character model"""
    model = CharacterModel(vocab_size)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    
    return model

def save_character_model(model, path: str):
    """Save character model to file"""
    torch.save(model.state_dict(), path)
