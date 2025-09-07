from typing import Dict, Any, List, Tuple
import random

class UpgradeDesigner:
    def __init__(self):
        # Upgrade templates for different weapon types
        self.upgrade_templates = {
            'assault_rifle': [
                {'name': 'Extended Magazine', 'type': 'magazine', 'effect': {'magazine_size': 10}, 'cost': 500},
                {'name': 'Improved Barrel', 'type': 'barrel', 'effect': {'damage': 5, 'accuracy': 5}, 'cost': 800},
                {'name': 'Recoil Reduction', 'type': 'stock', 'effect': {'control': 10}, 'cost': 600},
                {'name': 'Lightweight Stock', 'type': 'stock', 'effect': {'mobility': 10}, 'cost': 400},
                {'name': 'Optical Sight', 'type': 'sight', 'effect': {'accuracy': 15}, 'cost': 700},
                {'name': 'Suppressor', 'type': 'muzzle', 'effect': {'stealth': True}, 'cost': 900},
                {'name': 'Tactical Grip', 'type': 'underbarrel', 'effect': {'control': 8, 'accuracy': 5}, 'cost': 550}
            ],
            'sniper_rifle': [
                {'name': 'High-Power Scope', 'type': 'sight', 'effect': {'accuracy': 10, 'range': 100}, 'cost': 1000},
                {'name': 'Bipod', 'type': 'underbarrel', 'effect': {'control': 15}, 'cost': 600},
                {'name': 'Suppressor', 'type': 'muzzle', 'effect': {'stealth': True}, 'cost': 800},
                {'name': 'Extended Magazine', 'type': 'magazine', 'effect': {'magazine_size': 3}, 'cost': 500},
                {'name': 'Armor-Piercing Rounds', 'type': 'ammo', 'effect': {'damage': 20}, 'cost': 900},
                {'name': 'Lightweight Frame', 'type': 'stock', 'effect': {'mobility': 15}, 'cost': 700},
                {'name': 'Precision Barrel', 'type': 'barrel', 'effect': {'accuracy': 12}, 'cost': 850}
            ],
            'smg': [
                {'name': 'Extended Magazine', 'type': 'magazine', 'effect': {'magazine_size': 10}, 'cost': 400},
                {'name': 'Rapid Fire', 'type': 'barrel', 'effect': {'fire_rate': 200}, 'cost': 700},
                {'name': 'Laser Sight', 'type': 'underbarrel', 'effect': {'accuracy': 10}, 'cost': 500},
                {'name': 'Lightweight Frame', 'type': 'stock', 'effect': {'mobility': 10}, 'cost': 600},
                {'name': 'Suppressor', 'type': 'muzzle', 'effect': {'stealth': True}, 'cost': 800},
                {'name': 'Hollow Point Rounds', 'type': 'ammo', 'effect': {'damage': 8}, 'cost': 650},
                {'name': 'Vertical Grip', 'type': 'underbarrel', 'effect': {'control': 12}, 'cost': 550}
            ],
            'shotgun': [
                {'name': 'Extended Tube', 'type': 'magazine', 'effect': {'magazine_size': 4}, 'cost': 500},
                {'name': 'Choke', 'type': 'barrel', 'effect': {'accuracy': 15}, 'cost': 600},
                {'name': 'Slug Rounds', 'type': 'ammo', 'effect': {'damage': 20, 'range': 50}, 'cost': 800},
                {'name': 'Quick Pump', 'type': 'action', 'effect': {'fire_rate': 20}, 'cost': 700},
                {'name': 'Dragon\'s Breath', 'type': 'ammo', 'effect': {'fire_damage': True}, 'cost': 900},
                {'name': 'Tactical Stock', 'type': 'stock', 'effect': {'control': 10}, 'cost': 550},
                {'name': 'Short Barrel', 'type': 'barrel', 'effect': {'mobility': 15, 'range': -20}, 'cost': 450}
            ],
            'lmg': [
                {'name': 'Extended Belt', 'type': 'magazine', 'effect': {'magazine_size': 50}, 'cost': 600},
                {'name': 'Bipod', 'type': 'underbarrel', 'effect': {'control': 15}, 'cost': 500},
                {'name': 'Heavy Barrel', 'type': 'barrel', 'effect': {'damage': 10, 'accuracy': 8}, 'cost': 800},
                {'name': 'Quick Change', 'type': 'action', 'effect': {'reload_time': -1.0}, 'cost': 700},
                {'name': 'Armor-Piercing Rounds', 'type': 'ammo', 'effect': {'damage': 15}, 'cost': 900},
                {'name': 'Lightweight Frame', 'type': 'stock', 'effect': {'mobility': 12}, 'cost': 650},
                {'name': 'Compensator', 'type': 'muzzle', 'effect': {'control': 10}, 'cost': 550}
            ],
            'pistol': [
                {'name': 'Extended Magazine', 'type': 'magazine', 'effect': {'magazine_size': 5}, 'cost': 300},
                {'name': 'Match Grade Barrel', 'type': 'barrel', 'effect': {'accuracy': 10}, 'cost': 400},
                {'name': 'Hollow Point Rounds', 'type': 'ammo', 'effect': {'damage': 10}, 'cost': 500},
                {'name': 'Tactical Laser', 'type': 'underbarrel', 'effect': {'accuracy': 8}, 'cost': 350},
                {'name': 'Suppressor', 'type': 'muzzle', 'effect': {'stealth': True}, 'cost': 600},
                {'name': 'Lightweight Frame', 'type': 'frame', 'effect': {'mobility': 8}, 'cost': 400},
                {'name': 'Quick Draw', 'type': 'grip', 'effect': {'reload_time': -0.5}, 'cost': 450}
            ]
        }
        
        # Upgrade slot compatibility
        self.slot_compatibility = {
            'barrel': 1,
            'sight': 1,
            'muzzle': 1,
            'stock': 1,
            'underbarrel': 1,
            'magazine': 1,
            'ammo': 1,
            'action': 1,
            'frame': 1,
            'grip': 1
        }
        
        # Rarity modifiers for upgrade effects
        self.rarity_modifiers = {
            'common': 0.8,
            'uncommon': 1.0,
            'rare': 1.2,
            'epic': 1.5,
            'legendary': 2.0
        }
    
    def design_upgrade_path(self, weapon_type: str, rarity: str, 
                           num_upgrades: int = None) -> List[Dict[str, Any]]:
        """Design an upgrade path for a weapon"""
        if weapon_type not in self.upgrade_templates:
            raise ValueError(f"Unknown weapon type: {weapon_type}")
        
        if rarity not in self.rarity_modifiers:
            raise ValueError(f"Unknown rarity: {rarity}")
        
        # Determine number of upgrades based on rarity
        if num_upgrades is None:
            num_upgrades = {
                'common': 2,
                'uncommon': 3,
                'rare': 4,
                'epic': 5,
                'legendary': 6
            }.get(rarity, 3)
        
        # Get available upgrades for this weapon type
        available_upgrades = self.upgrade_templates[weapon_type].copy()
        
        # Select upgrades
        selected_upgrades = []
        used_slots = {slot: 0 for slot in self.slot_compatibility}
        
        # Always include at least one damage upgrade for higher rarities
        if rarity in ['rare', 'epic', 'legendary']:
            damage_upgrades = [u for u in available_upgrades if 'damage' in u.get('effect', {})]
            if damage_upgrades:
                upgrade = random.choice(damage_upgrades)
                selected_upgrades.append(self.apply_rarity_modifier(upgrade, rarity))
                available_upgrades.remove(upgrade)
                
                # Mark slot as used
                slot = upgrade['type']
                used_slots[slot] = used_slots.get(slot, 0) + 1
        
        # Select remaining upgrades
        while len(selected_upgrades) < num_upgrades and available_upgrades:
            # Filter upgrades by available slots
            valid_upgrades = [
                u for u in available_upgrades 
                if used_slots.get(u['type'], 0) < self.slot_compatibility.get(u['type'], 1)
            ]
            
            if not valid_upgrades:
                break
                
            # Select a random upgrade
            upgrade = random.choice(valid_upgrades)
            selected_upgrades.append(self.apply_rarity_modifier(upgrade, rarity))
            available_upgrades.remove(upgrade)
            
            # Mark slot as used
            slot = upgrade['type']
            used_slots[slot] = used_slots.get(slot, 0) + 1
        
        return selected_upgrades
    
    def apply_rarity_modifier(self, upgrade: Dict[str, Any], rarity: str) -> Dict[str, Any]:
        """Apply rarity modifier to upgrade effects"""
        modifier = self.rarity_modifiers.get(rarity, 1.0)
        
        # Create a copy of the upgrade
        modified_upgrade = upgrade.copy()
        
        # Modify numeric effects
        if 'effect' in modified_upgrade:
            modified_effect = {}
            for stat, value in modified_upgrade['effect'].items():
                if isinstance(value, (int, float)):
                    modified_effect[stat] = int(value * modifier)
                else:
                    modified_effect[stat] = value
            
            modified_upgrade['effect'] = modified_effect
        
        # Modify cost
        if 'cost' in modified_upgrade:
            modified_upgrade['cost'] = int(modified_upgrade['cost'] * modifier)
        
        return modified_upgrade
    
    def validate_upgrade_compatibility(self, upgrades: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate if upgrades are compatible with each other"""
        slot_counts = {}
        issues = []
        
        for upgrade in upgrades:
            slot = upgrade['type']
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
            
            # Check if slot is over capacity
            if slot_counts[slot] > self.slot_compatibility.get(slot, 1):
                issues.append(f"Too many {slot} upgrades: {slot_counts[slot]}")
        
        # Check for incompatible combinations
        has_suppressor = any(u.get('name') == 'Suppressor' for u in upgrades)
        has_compensator = any(u.get('name') == 'Compensator' for u in upgrades)
        
        if has_suppressor and has_compensator:
            issues.append("Cannot have both Suppressor and Compensator")
        
        return len(issues) == 0, issues
    
    def calculate_upgrade_effects(self, base_stats: Dict[str, float], 
                                upgrades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the combined effects of upgrades on base stats"""
        result_stats = base_stats.copy()
        
        for upgrade in upgrades:
            effects = upgrade.get('effect', {})
            
            for stat, value in effects.items():
                if stat in result_stats:
                    if stat == 'reload_time':
                        # For reload time, lower is better (subtract)
                        result_stats[stat] = max(0.5, result_stats[stat] - value * 0.1)
                    else:
                        result_stats[stat] += value
                else:
                    # Handle special stats
                    result_stats[stat] = value
        
        return result_stats
    
    def generate_upgrade_tree(self, weapon_type: str, rarity: str, 
                             depth: int = 3) -> Dict[str, Any]:
        """Generate a branching upgrade tree for a weapon"""
        # Get all possible upgrades
        all_upgrades = self.upgrade_templates.get(weapon_type, []).copy()
        
        # Apply rarity modifiers
        upgraded_all = [self.apply_rarity_modifier(u, rarity) for u in all_upgrades]
        
        # Create tree structure
        tree = {
            'tier_1': [],
            'tier_2': [],
            'tier_3': []
        }
        
        # Select upgrades for each tier
        for tier in ['tier_1', 'tier_2', 'tier_3']:
            num_upgrades = 3 if tier == 'tier_1' else 2
            
            # Filter upgrades that are compatible with already selected ones
            compatible_upgrades = []
            for upgrade in upgraded_all:
                # Check if this upgrade is already in the tree
                already_in_tree = any(
                    u['name'] == upgrade['name'] 
                    for t in tree.values() 
                    for u in t
                )
                
                if not already_in_tree:
                    compatible_upgrades.append(upgrade)
            
            # Select upgrades for this tier
            selected = random.sample(compatible_upgrades, min(num_upgrades, len(compatible_upgrades)))
            tree[tier] = selected
        
        return tree
    
    def optimize_upgrade_path(self, base_stats: Dict[str, float], 
                            available_upgrades: List[Dict[str, Any]],
                            target_stats: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find the optimal upgrade path to achieve target stats"""
        # This is a simplified implementation
        # In a real scenario, this would use more advanced optimization algorithms
        
        best_path = []
        best_score = float('inf')
        
        # Try different combinations (limited to avoid combinatorial explosion)
        for i in range(100):  # Limit iterations
            # Randomly select some upgrades
            num_upgrades = random.randint(1, min(5, len(available_upgrades)))
            selected = random.sample(available_upgrades, num_upgrades)
            
            # Calculate resulting stats
            result_stats = self.calculate_upgrade_effects(base_stats, selected)
            
            # Calculate distance from target stats
            score = 0
            for stat, target_value in target_stats.items():
                if stat in result_stats:
                    score += abs(result_stats[stat] - target_value)
                else:
                    score += target_value  # Penalty for missing stats
            
            # Update best path if this is better
            if score < best_score:
                best_score = score
                best_path = selected
        
        return best_path
    
    def create_synergistic_upgrades(self, weapon_type: str, theme: str = None) -> List[Dict[str, Any]]:
        """Create upgrades that synergize with each other"""
        base_upgrades = self.upgrade_templates.get(weapon_type, []).copy()
        
        # Define synergy themes
        themes = {
            'stealth': ['Suppressor', 'Extended Magazine', 'Optical Sight'],
            'assault': ['Extended Magazine', 'Improved Barrel', 'Recoil Reduction'],
            'precision': ['High-Power Scope', 'Bipod', 'Precision Barrel'],
            'mobility': ['Lightweight Stock', 'Short Barrel', 'Quick Draw'],
            'capacity': ['Extended Magazine', 'Extended Belt', 'Quick Change']
        }
        
        # Select theme or choose random
        if theme is None:
            theme = random.choice(list(themes.keys()))
        
        # Filter upgrades that match the theme
        thematic_upgrades = []
        for upgrade in base_upgrades:
            if upgrade['name'] in themes[theme]:
                thematic_upgrades.append(upgrade)
        
        # Add some random upgrades to fill out the set
        other_upgrades = [u for u in base_upgrades if u not in thematic_upgrades]
        num_additional = min(3, len(other_upgrades))
        
        if other_upgrades and num_additional > 0:
            additional = random.sample(other_upgrades, num_additional)
            thematic_upgrades.extend(additional)
        
        return thematic_upgrades

# Example usage
if __name__ == "__main__":
    designer = UpgradeDesigner()
    
    # Design an upgrade path for an assault rifle
    upgrades = designer.design_upgrade_path("assault_rifle", "rare")
    print("Assault Rifle upgrades:", upgrades)
    
    # Validate upgrade compatibility
    is_valid, issues = designer.validate_upgrade_compatibility(upgrades)
    print(f"Upgrades valid: {is_valid}, Issues: {issues}")
    
    # Calculate upgrade effects
    base_stats = {
        'damage': 25,
        'fire_rate': 600,
        'accuracy': 70,
        'range': 300,
        'mobility': 70,
        'control': 60,
        'magazine_size': 30,
        'reload_time': 2.5
    }
    
    result_stats = designer.calculate_upgrade_effects(base_stats, upgrades)
    print("Base stats:", base_stats)
    print("Upgraded stats:", result_stats)
    
    # Generate an upgrade tree
    upgrade_tree = designer.generate_upgrade_tree("sniper_rifle", "epic")
    print("Upgrade tree:", upgrade_tree)
    
    # Create synergistic upgrades
    synergistic = designer.create_synergistic_upgrades("smg", "mobility")
    print("Synergistic upgrades:", synergistic)
