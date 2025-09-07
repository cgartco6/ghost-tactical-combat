import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CharacterTrainingExample:
    description: str
    traits: List[str]
    skills: Dict[str, float]
    type: str
    specialization: str

class CharacterDataset:
    def __init__(self, data_path: str = "data/training/characters.json"):
        self.data_path = Path(data_path)
        self.examples = []
        self.load_data()
    
    def load_data(self):
        """Load training data from file"""
        if not self.data_path.exists():
            return
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            example = CharacterTrainingExample(
                description=item['description'],
                traits=item['traits'],
                skills=item['skills'],
                type=item['type'],
                specialization=item['specialization']
            )
            self.examples.append(example)
    
    def save_data(self):
        """Save training data to file"""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for example in self.examples:
            data.append({
                'description': example.description,
                'traits': example.traits,
                'skills': example.skills,
                'type': example.type,
                'specialization': example.specialization
            })
        
        with open(self.data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_example(self, example: CharacterTrainingExample):
        """Add a new training example"""
        self.examples.append(example)
    
    def get_examples_by_type(self, char_type: str) -> List[CharacterTrainingExample]:
        """Get examples filtered by character type"""
        return [ex for ex in self.examples if ex.type == char_type]
    
    def get_examples_by_specialization(self, specialization: str) -> List[CharacterTrainingExample]:
        """Get examples filtered by specialization"""
        return [ex for ex in self.examples if ex.specialization == specialization]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_training_example_from_character(character_data: Dict[str, Any]) -> CharacterTrainingExample:
    """Create a training example from character data"""
    return CharacterTrainingExample(
        description=character_data['personality']['description'],
        traits=character_data['personality']['traits'],
        skills=character_data['skills'],
        type=character_data['concept']['type'],
        specialization=character_data['concept']['specialization']
    )
