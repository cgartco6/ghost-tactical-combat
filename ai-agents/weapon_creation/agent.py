import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from typing import Dict, Any, List
import asyncio

class WeaponAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize AI models for weapon creation"""
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model_loaded = True
            print("Weapon creation models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def design_weapon(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Design a new weapon"""
        self.current_task = "design_weapon"
        
        try:
            weapon_type = params.get('type', 'assault_rifle')
            rarity = params.get('rarity', 'common')
            
            # Generate weapon concept
            concept = await self.generate_weapon_concept(weapon_type, rarity)
            
            # Generate visual assets
            assets = await self.generate_weapon_assets(concept)
            
            # Generate stats
            stats = await self.generate_weapon_stats(concept)
            
            # Generate upgrade path
            upgrades = await self.generate_upgrade_path(concept)
            
            return {
                'success': True,
                'weapon_id': f"wpn_{weapon_type}_{hash(str(concept))}",
                'concept': concept,
                'assets': assets,
                'stats': stats,
                'upgrades': upgrades
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_weapon_concept(self, weapon_type: str, rarity: str) -> Dict[str, Any]:
        """Generate weapon concept"""
        weapon_types = {
            'assault_rifle': {
                'description': 'A versatile assault rifle for medium-range combat',
                'fire_modes': ['single', 'burst', 'auto'],
                'caliber': '5.56mm'
            },
            'sniper_rifle': {
                'description': 'A high-precision sniper rifle for long-range engagements',
                'fire_modes': ['single'],
                'caliber': '7.62mm'
            },
            'smg': {
                'description': 'A compact submachine gun for close-quarters combat',
                'fire_modes': ['single', 'auto'],
                'caliber': '9mm'
            },
            'shotgun': {
                'description': 'A powerful shotgun for close-range engagements',
                'fire_modes': ['pump_action', 'semi_auto'],
                'caliber': '12 gauge'
            }
        }
        
        rarities = {
            'common': {'color': '#a0a0a0', 'modifier': 1.0},
            'uncommon': {'color': '#00ff00', 'modifier': 1.2},
            'rare': {'color': '#0080ff', 'modifier': 1.5},
            'epic': {'color': '#8000ff', 'modifier': 2.0},
            'legendary': {'color': '#ff8000', 'modifier': 3.0}
        }
        
        concept = weapon_types.get(weapon_type, weapon_types['assault_rifle']).copy()
        concept['type'] = weapon_type
        concept['rarity'] = rarity
        concept['rarity_info'] = rarities[rarity]
        concept['name'] = self.generate_weapon_name(weapon_type, rarity)
        
        return concept
    
    def generate_weapon_name(self, weapon_type: str, rarity: str) -> str:
        """Generate a name for the weapon"""
        prefixes = {
            'common': ['Standard', 'Basic', 'Standard Issue'],
            'uncommon': ['Tactical', 'Advanced', 'Enhanced'],
            'rare': ['Precision', 'Combat', 'Elite'],
            'epic': ['Superior', 'Advanced Combat', 'Tactical Elite'],
            'legendary': ['Legendary', 'Ultimate', 'Omega']
        }
        
        bases = {
            'assault_rifle': ['Assault Rifle', 'Battle Rifle', 'Combat Rifle'],
            'sniper_rifle': ['Sniper Rifle', 'Long Rifle', 'Precision Rifle'],
            'smg': ['SMG', 'Submachine Gun', 'Compact SMG'],
            'shotgun': ['Shotgun', 'Combat Shotgun', 'Tactical Shotgun']
        }
        
        prefix = np.random.choice(prefixes[rarity])
        base = np.random.choice(bases[weapon_type])
        
        return f"{prefix} {base}"
    
    async def generate_weapon_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual assets for the weapon"""
        if not self.model_loaded:
            return {'error': 'Models not loaded'}
        
        assets = {}
        weapon_type = concept['type']
        rarity = concept['rarity']
        
        # Generate weapon texture
        texture_prompt = f"4k texture for a {rarity} {weapon_type}, photorealistic, game asset"
        with torch.no_grad():
            texture = self.sd_pipeline(texture_prompt).images[0]
        
        texture_path = f"assets/weapons/{weapon_type}/textures/{concept['name'].replace(' ', '_')}_diffuse.png"
        texture.save(texture_path)
        assets['texture'] = texture_path
        
        # Generate normal map
        normal_prompt = f"normal map for a {weapon_type}, game asset"
        with torch.no_grad():
            normal_map = self.sd_pipeline(normal_prompt).images[0]
        
        normal_path = f"assets/weapons/{weapon_type}/textures/{concept['name'].replace(' ', '_')}_normal.png"
        normal_map.save(normal_path)
        assets['normal_map'] = normal_path
        
        # Generate model (placeholder - would use a 3D model generator)
        assets['model'] = f"models/weapons/{weapon_type}/{concept['name'].replace(' ', '_')}.fbx"
        
        return assets
    
    async def generate_weapon_stats(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stats for the weapon"""
        base_stats = {
            'assault_rifle': {
                'damage': 25,
                'fire_rate': 600,
                'accuracy': 70,
                'range': 300,
                'mobility': 70,
                'magazine_size': 30
            },
            'sniper_rifle': {
                'damage': 100,
                'fire_rate': 40,
                'accuracy': 95,
                'range': 1000,
                'mobility': 40,
                'magazine_size': 5
            },
            'smg': {
                'damage': 20,
                'fire_rate': 800,
                'accuracy': 60,
                'range': 150,
                'mobility': 85,
                'magazine_size': 25
            },
            'shotgun': {
                'damage': 80,  # per pellet
                'fire_rate': 70,
                'accuracy': 50,  # spread-based
                'range': 50,
                'mobility': 60,
                'magazine_size': 8
            }
        }
        
        # Get base stats for weapon type
        stats = base_stats.get(concept['type'], base_stats['assault_rifle']).copy()
        
        # Apply rarity modifier
        rarity_modifier = concept['rarity_info']['modifier']
        for stat in stats:
            if stat != 'magazine_size':  # Don't modify magazine size with rarity
                stats[stat] = int(stats[stat] * rarity_modifier)
        
        return stats
    
    async def generate_upgrade_path(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate upgrade path for the weapon"""
        upgrades = []
        weapon_type = concept['type']
        
        # Define possible upgrades based on weapon type
        upgrade_options = {
            'assault_rifle': [
                {'name': 'Extended Magazine', 'cost': 500, 'effect': {'magazine_size': 10}},
                {'name': 'Improved Barrel', 'cost': 800, 'effect': {'damage': 5, 'accuracy': 5}},
                {'name': 'Recoil Reduction', 'cost': 600, 'effect': {'accuracy': 10}},
                {'name': 'Lightweight Stock', 'cost': 400, 'effect': {'mobility': 10}}
            ],
            'sniper_rifle': [
                {'name': 'High-Power Scope', 'cost': 1000, 'effect': {'accuracy': 10, 'range': 100}},
                {'name': 'Bipod', 'cost': 600, 'effect': {'accuracy': 15}},
                {'name': 'Suppressor', 'cost': 800, 'effect': {'stealth': True}},
                {'name': 'Extended Magazine', 'cost': 500, 'effect': {'magazine_size': 3}}
            ],
            'smg': [
                {'name': 'Extended Magazine', 'cost': 400, 'effect': {'magazine_size': 10}},
                {'name': 'Rapid Fire', 'cost': 700, 'effect': {'fire_rate': 200}},
                {'name': 'Laser Sight', 'cost': 500, 'effect': {'accuracy': 10}},
                {'name': 'Lightweight Frame', 'cost': 600, 'effect': {'mobility': 10}}
            ],
            'shotgun': [
                {'name': 'Extended Tube', 'cost': 500, 'effect': {'magazine_size': 4}},
                {'name': 'Choke', 'cost': 600, 'effect': {'accuracy': 15}},
                {'name': 'Slug Rounds', 'cost': 800, 'effect': {'damage': 20, 'range': 50}},
                {'name': 'Quick Pump', 'cost': 700, 'effect': {'fire_rate': 20}}
            ]
        }
        
        # Select 3-4 upgrades for this weapon
        options = upgrade_options.get(weapon_type, upgrade_options['assault_rifle'])
        selected_upgrades = np.random.choice(options, size=min(4, len(options)), replace=False)
        
        for upgrade in selected_upgrades:
            upgrades.append(upgrade)
        
        return upgrades
    
    def get_status(self) -> str:
        """Get current agent status"""
        if not self.model_loaded:
            return "offline"
        return "busy" if self.current_task else "idle"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "weapon_concept_generation",
            "weapon_stat_balancing",
            "visual_asset_creation",
            "upgrade_path_design"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        torch.cuda.empty_cache()
