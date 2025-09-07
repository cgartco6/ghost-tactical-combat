import json
from pathlib import Path
from typing import List, Dict, Any
from .trainer import CharacterTrainingDataset
from ..models.training_data import CharacterDataset

def load_character_training_data(data_dir: str = "data/training") -> CharacterDataset:
    """Load character training data from directory"""
    dataset = CharacterDataset()
    
    # Load additional data from files
    data_files = [
        "special_forces.json",
        "soldiers.json",
        "operatives.json"
    ]
    
    for file_name in data_files:
        file_path = Path(data_dir) / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                example = CharacterTrainingExample(
                    description=item['description'],
                    traits=item['traits'],
                    skills=item['skills'],
                    type=item['type'],
                    specialization=item['specialization']
                )
                dataset.add_example(example)
    
    return dataset

def augment_training_data(dataset: CharacterDataset, augmentation_factor: int = 2) -> CharacterDataset:
    """Augment training data by creating variations"""
    augmented = CharacterDataset()
    
    for example in dataset.examples:
        # Add original example
        augmented.add_example(example)
        
        # Create augmented variations
        for i in range(augmentation_factor):
            # Create variation by slightly modifying description
            words = example.description.split()
            if len(words) > 5:
                # Swap two random words
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                
                augmented_desc = " ".join(words)
                
                # Create augmented example
                augmented_example = CharacterTrainingExample(
                    description=augmented_desc,
                    traits=example.traits,
                    skills=example.skills,
                    type=example.type,
                    specialization=example.specialization
                )
                
                augmented.add_example(augmented_example)
    
    return augmented

def export_training_data(dataset: CharacterDataset, export_path: str):
    """Export training data to file"""
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for example in dataset.examples:
        data.append({
            'description': example.description,
            'traits': example.traits,
            'skills': example.skills,
            'type': example.type,
            'specialization': example.specialization
        })
    
    with open(export_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_predefined_character_templates():
    """Create predefined character templates for training"""
    templates = [
        {
            'type': 'special_forces',
            'specialization': 'assault',
            'traits': ['confident', 'aggressive', 'loyal'],
            'skills': {
                'marksmanship': 85,
                'stealth': 70,
                'tactics': 75,
                'endurance': 90,
                'technical': 60,
                'medical': 50,
                'explosives': 75,
                'leadership': 70
            }
        },
        {
            'type': 'special_forces',
            'specialization': 'sniper',
            'traits': ['calm', 'cautious', 'stoic'],
            'skills': {
                'marksmanship': 95,
                'stealth': 90,
                'tactics': 80,
                'endurance': 85,
                'technical': 70,
                'medical': 60,
                'explosives': 65,
                'leadership': 60
            }
        },
        {
            'type': 'special_forces',
            'specialization': 'engineer',
            'traits': ['intelligent', 'cautious', 'calm'],
            'skills': {
                'marksmanship': 70,
                'stealth': 65,
                'tactics': 75,
                'endurance': 75,
                'technical': 95,
                'medical': 60,
                'explosives': 85,
                'leadership': 65
            }
        }
    ]
    
    return templates
