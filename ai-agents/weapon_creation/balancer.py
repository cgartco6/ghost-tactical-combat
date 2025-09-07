import numpy as np
from typing import Dict, Any, List, Tuple
import random

class WeaponBalancer:
    def __init__(self):
        # Base stats for different weapon types
        self.base_stats = {
            'assault_rifle': {
                'damage': 25,
                'fire_rate': 600,
                'accuracy': 70,
                'range': 300,
                'mobility': 70,
                'control': 60,
                'magazine_size': 30,
                'reload_time': 2.5
            },
            'sniper_rifle': {
                'damage': 100,
                'fire_rate': 40,
                'accuracy': 95,
                'range': 1000,
                'mobility': 40,
                'control': 50,
                'magazine_size': 5,
                'reload_time': 3.5
            },
            'smg': {
                'damage': 20,
                'fire_rate': 800,
                'accuracy': 60,
                'range': 150,
                'mobility': 85,
                'control': 40,
                'magazine_size': 25,
                'reload_time': 2.0
            },
            'shotgun': {
                'damage': 80,  # per pellet
                'fire_rate': 70,
                'accuracy': 50,  # spread-based
                'range': 50,
                'mobility': 60,
                'control': 30,
                'magazine_size': 8,
                'reload_time': 4.0
            },
            'lmg': {
                'damage': 30,
                'fire_rate': 700,
                'accuracy': 65,
                'range': 400,
                'mobility': 50,
                'control': 35,
                'magazine_size': 100,
                'reload_time': 5.0
            },
            'pistol': {
                'damage': 35,
                'fire_rate': 300,
                'accuracy': 75,
                'range': 100,
                'mobility': 90,
                'control': 70,
                'magazine_size': 12,
                'reload_time': 1.5
            }
        }
        
        # Rarity modifiers
        self.rarity_modifiers = {
            'common': 1.0,
            'uncommon': 1.1,
            'rare': 1.25,
            'epic': 1.5,
            'legendary': 2.0
        }
        
        # Stat weights for balance calculation
        self.stat_weights = {
            'damage': 1.2,
            'fire_rate': 1.0,
            'accuracy': 0.8,
            'range': 0.7,
            'mobility': 0.6,
            'control': 0.5,
            'magazine_size': 0.4,
            'reload_time': -0.5  # Negative because lower is better
        }
    
    def calculate_weapon_score(self, stats: Dict[str, float]) -> float:
        """Calculate a balance score for a weapon"""
        score = 0
        
        for stat, value in stats.items():
            if stat in self.stat_weights:
                # Normalize value to 0-100 scale
                normalized_value = value
                
                # Handle special cases
                if stat == 'reload_time':
                    # Lower reload time is better, so invert
                    normalized_value = 10 - min(value, 10)
                
                score += normalized_value * self.stat_weights[stat]
        
        return score
    
    def balance_weapon_stats(self, weapon_type: str, rarity: str, 
                            target_score: float = None) -> Dict[str, float]:
        """Generate balanced stats for a weapon"""
        if weapon_type not in self.base_stats:
            raise ValueError(f"Unknown weapon type: {weapon_type}")
        
        if rarity not in self.rarity_modifiers:
            raise ValueError(f"Unknown rarity: {rarity}")
        
        # Get base stats
        base_stats = self.base_stats[weapon_type].copy()
        modifier = self.rarity_modifiers[rarity]
        
        # Apply rarity modifier
        stats = {}
        for stat, value in base_stats.items():
            if stat == 'reload_time':
                # Lower reload time is better, so divide
                stats[stat] = max(0.5, value / modifier)
            elif stat == 'fire_rate':
                # Fire rate is inverse (higher is better but shouldn't scale linearly)
                stats[stat] = int(value * (2 - modifier))
            else:
                stats[stat] = int(value * modifier)
        
        # If target score is provided, adjust to match
        if target_score is not None:
            current_score = self.calculate_weapon_score(stats)
            
            # Adjust stats to match target score
            if current_score < target_score:
                # Increase stats
                stats = self.increase_weapon_score(stats, target_score - current_score)
            elif current_score > target_score:
                # Decrease stats
                stats = self.decrease_weapon_score(stats, current_score - target_score)
        
        return stats
    
    def increase_weapon_score(self, stats: Dict[str, float], amount: float) -> Dict[str, float]:
        """Increase weapon score by the specified amount"""
        # Calculate how much each stat contributes to the score
        contributions = {}
        total_contribution = 0
        
        for stat, value in stats.items():
            if stat in self.stat_weights:
                contribution = value * self.stat_weights[stat]
                contributions[stat] = contribution
                total_contribution += contribution
        
        # Increase stats proportionally to their current contribution
        result = stats.copy()
        
        for stat, contribution in contributions.items():
            if total_contribution > 0:
                increase_factor = contribution / total_contribution
                increase_amount = amount * increase_factor / self.stat_weights[stat]
                
                if stat == 'reload_time':
                    # For reload time, decrease instead of increase
                    result[stat] = max(0.5, result[stat] - increase_amount * 0.1)
                else:
                    result[stat] += increase_amount
        
        return result
    
    def decrease_weapon_score(self, stats: Dict[str, float], amount: float) -> Dict[str, float]:
        """Decrease weapon score by the specified amount"""
        # Calculate how much each stat contributes to the score
        contributions = {}
        total_contribution = 0
        
        for stat, value in stats.items():
            if stat in self.stat_weights:
                contribution = value * self.stat_weights[stat]
                contributions[stat] = contribution
                total_contribution += contribution
        
        # Decrease stats proportionally to their current contribution
        result = stats.copy()
        
        for stat, contribution in contributions.items():
            if total_contribution > 0:
                decrease_factor = contribution / total_contribution
                decrease_amount = amount * decrease_factor / self.stat_weights[stat]
                
                if stat == 'reload_time':
                    # For reload time, increase instead of decrease
                    result[stat] = result[stat] + decrease_amount * 0.1
                else:
                    result[stat] = max(1, result[stat] - decrease_amount)
        
        return result
    
    def get_target_score_for_rarity(self, rarity: str) -> float:
        """Get target balance score for a rarity level"""
        base_score = 100  # Common weapons
        
        rarity_scores = {
            'common': base_score,
            'uncommon': base_score * 1.2,
            'rare': base_score * 1.5,
            'epic': base_score * 2.0,
            'legendary': base_score * 3.0
        }
        
        return rarity_scores.get(rarity, base_score)
    
    def generate_balanced_weapon(self, weapon_type: str, rarity: str) -> Dict[str, float]:
        """Generate a fully balanced weapon"""
        target_score = self.get_target_score_for_rarity(rarity)
        return self.balance_weapon_stats(weapon_type, rarity, target_score)
    
    def validate_weapon_balance(self, stats: Dict[str, float], 
                               weapon_type: str, rarity: str) -> Tuple[bool, float, float]:
        """Validate if a weapon is balanced for its type and rarity"""
        actual_score = self.calculate_weapon_score(stats)
        target_score = self.get_target_score_for_rarity(rarity)
        
        # Allow 10% variance
        min_score = target_score * 0.9
        max_score = target_score * 1.1
        
        is_balanced = min_score <= actual_score <= max_score
        return is_balanced, actual_score, target_score
    
    def adjust_stat_for_balance(self, stats: Dict[str, float], stat_name: str, 
                               adjustment: float) -> Dict[str, float]:
        """Adjust a specific stat and rebalance other stats to maintain overall balance"""
        if stat_name not in stats:
            return stats
        
        # Calculate current score
        current_score = self.calculate_weapon_score(stats)
        
        # Adjust the specified stat
        new_stats = stats.copy()
        
        if stat_name == 'reload_time':
            # For reload time, lower is better
            new_stats[stat_name] = max(0.5, new_stats[stat_name] + adjustment)
        else:
            new_stats[stat_name] = max(1, new_stats[stat_name] + adjustment)
        
        # Calculate new score
        new_score = self.calculate_weapon_score(new_stats)
        
        # Rebalance other stats to maintain overall balance
        if new_score > current_score:
            # Need to decrease other stats
            return self.decrease_weapon_score(new_stats, new_score - current_score)
        else:
            # Need to increase other stats
            return self.increase_weapon_score(new_stats, current_score - new_score)
    
    def create_weapon_class_balance(self, weapon_classes: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Create a balanced set of weapons across different classes"""
        if weapon_classes is None:
            weapon_classes = list(self.base_stats.keys())
        
        balanced_weapons = {}
        
        for weapon_class in weapon_classes:
            balanced_weapons[weapon_class] = {}
            
            for rarity in self.rarity_modifiers.keys():
                balanced_weapons[weapon_class][rarity] = self.generate_balanced_weapon(weapon_class, rarity)
        
        return balanced_weapons

# Example usage
if __name__ == "__main__":
    balancer = WeaponBalancer()
    
    # Generate a balanced assault rifle
    ar_stats = balancer.generate_balanced_weapon("assault_rifle", "rare")
    print("Assault Rifle (Rare) stats:", ar_stats)
    
    # Check balance
    is_balanced, actual_score, target_score = balancer.validate_weapon_balance(ar_stats, "assault_rifle", "rare")
    print(f"Balanced: {is_balanced}, Score: {actual_score:.2f}, Target: {target_score:.2f}")
    
    # Create a balanced set of all weapon classes
    balanced_set = balancer.create_weapon_class_balance()
    print("Balanced weapon set created for all classes and rarities")
    
    # Adjust a specific stat and rebalance
    adjusted_stats = balancer.adjust_stat_for_balance(ar_stats, "damage", 10)
    print("After increasing damage by 10:", adjusted_stats)
    
    # Check new balance
    is_balanced, actual_score, target_score = balancer.validate_weapon_balance(adjusted_stats, "assault_rifle", "rare")
    print(f"Balanced: {is_balanced}, Score: {actual_score:.2f}, Target: {target_score:.2f}")
