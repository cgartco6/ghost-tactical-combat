import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from typing import Dict, Any, List
import asyncio

class EnvironmentAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize AI models for environment generation"""
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model_loaded = True
            print("Environment generation models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def generate_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a game environment"""
        self.current_task = "generate_environment"
        
        try:
            env_type = params.get('type', 'jungle')
            mission_type = params.get('mission_type', 'infiltration')
            
            # Generate environment concept
            concept = await self.generate_environment_concept(env_type, mission_type)
            
            # Generate visual assets
            assets = await self.generate_environment_assets(concept)
            
            # Generate terrain data
            terrain = await self.generate_terrain(concept)
            
            # Generate props and interactive elements
            props = await self.generate_props(concept, mission_type)
            
            return {
                'success': True,
                'environment_id': f"env_{env_type}_{hash(str(concept))}",
                'concept': concept,
                'assets': assets,
                'terrain': terrain,
                'props': props
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_environment_concept(self, env_type: str, mission_type: str) -> Dict[str, Any]:
        """Generate environment concept"""
        themes = {
            'jungle': {
                'description': 'Dense tropical jungle with ancient ruins',
                'colors': ['#2d5a27', '#3a7a3a', '#4d8c4d', '#63a63a'],
                'weather': ['rain', 'fog', 'sunny']
            },
            'urban': {
                'description': 'War-torn urban environment with destroyed buildings',
                'colors': ['#555555', '#777777', '#999999', '#aaaaaa'],
                'weather': ['clear', 'rain', 'dust_storm']
            },
            'desert': {
                'description': 'Arid desert with sand dunes and rocky outcrops',
                'colors': ['#d9bb7b', '#e3c28b', '#ecd19d', '#f5e0b1'],
                'weather': ['sunny', 'heat_haze', 'sandstorm']
            },
            'arctic': {
                'description': 'Snow-covered arctic landscape with icy mountains',
                'colors': ['#ffffff', '#e6e6e6', '#cccccc', '#b3b3b3'],
                'weather': ['snow', 'blizzard', 'clear_cold']
            }
        }
        
        concept = themes.get(env_type, themes['jungle']).copy()
        concept['type'] = env_type
        concept['mission_type'] = mission_type
        concept['lighting'] = self.generate_lighting(env_type, mission_type)
        
        return concept
    
    def generate_lighting(self, env_type: str, mission_type: str) -> Dict[str, Any]:
        """Generate lighting settings for the environment"""
        lighting_profiles = {
            'jungle': {
                'day': {'intensity': 1.0, 'color': '#ffebc1', 'ambient': '#3a5a2a'},
                'night': {'intensity': 0.2, 'color': '#4466aa', 'ambient': '#0a1a2a'},
                'dusk': {'intensity': 0.6, 'color': '#ff8855', 'ambient': '#2a3a4a'}
            },
            'urban': {
                'day': {'intensity': 1.0, 'color': '#ffdbac', 'ambient': '#555555'},
                'night': {'intensity': 0.3, 'color': '#4466aa', 'ambient': '#222222'},
                'dusk': {'intensity': 0.7, 'color': '#ff7744', 'ambient': '#333333'}
            }
        }
        
        # Default to jungle if environment type not found
        profile = lighting_profiles.get(env_type, lighting_profiles['jungle'])
        
        # Select lighting based on mission type
        if mission_type == 'stealth':
            return profile['night']
        elif mission_type == 'assault':
            return profile['day']
        else:
            return profile['dusk']
    
    async def generate_environment_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual assets for the environment"""
        if not self.model_loaded:
            return {'error': 'Models not loaded'}
        
        assets = {}
        env_type = concept['type']
        
        # Generate terrain textures
        terrain_prompt = f"4k texture for {concept['description']}, photorealistic, game asset"
        with torch.no_grad():
            terrain_texture = self.sd_pipeline(terrain_prompt).images[0]
        
        terrain_path = f"assets/environments/{env_type}/textures/terrain_diffuse.png"
        terrain_texture.save(terrain_path)
        assets['terrain_texture'] = terrain_path
        
        # Generate additional textures based on environment type
        if env_type == 'jungle':
            assets['foliage'] = await self.generate_jungle_assets(concept)
        elif env_type == 'urban':
            assets['buildings'] = await self.generate_urban_assets(concept)
        elif env_type == 'desert':
            assets['rock_formations'] = await self.generate_desert_assets(concept)
        elif env_type == 'arctic':
            assets['ice_formations'] = await self.generate_arctic_assets(concept)
        
        return assets
    
    async def generate_jungle_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate jungle-specific assets"""
        assets = {}
        
        # Generate foliage textures
        foliage_prompts = [
            "4k tropical leaves texture, photorealistic",
            "4k jungle tree bark texture, photorealistic",
            "4k jungle ground cover texture, photorealistic"
        ]
        
        for i, prompt in enumerate(foliage_prompts):
            with torch.no_grad():
                texture = self.sd_pipeline(prompt).images[0]
            path = f"assets/environments/jungle/textures/foliage_{i}.png"
            texture.save(path)
            assets[f'foliage_{i}'] = path
        
        return assets
    
    async def generate_terrain(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate terrain data"""
        env_type = concept['type']
        
        # Generate heightmap
        heightmap = self.generate_heightmap(env_type)
        heightmap_path = f"assets/environments/{env_type}/terrain/heightmap.png"
        heightmap.save(heightmap_path)
        
        # Generate splatmap for texture blending
        splatmap = self.generate_splatmap(env_type)
        splatmap_path = f"assets/environments/{env_type}/terrain/splatmap.png"
        splatmap.save(splatmap_path)
        
        return {
            'heightmap': heightmap_path,
            'splatmap': splatmap_path,
            'size': [1024, 1024],  # terrain size in meters
            'max_height': 256 if env_type == 'mountain' else 128  # max height in meters
        }
    
    def generate_heightmap(self, env_type: str):
        """Generate a heightmap for the terrain"""
        # This would use more sophisticated terrain generation
        # For now, return a simple placeholder
        from PIL import Image
        import numpy as np
        
        size = 512
        if env_type == 'mountain':
            # Mountainous terrain
            heightmap = np.random.rand(size, size) * 255
        else:
            # Relatively flat terrain with some variation
            heightmap = np.random.rand(size, size) * 128
        
        return Image.fromarray(heightmap.astype(np.uint8))
    
    def generate_splatmap(self, env_type: str):
        """Generate a splatmap for texture blending"""
        from PIL import Image
        import numpy as np
        
        size = 512
        # Create RGB splatmap where each channel represents a different texture
        splatmap = np.zeros((size, size, 3), dtype=np.uint8)
        
        if env_type == 'jungle':
            # Green channel for grass, red for dirt, blue for rock
            splatmap[:, :, 1] = 255  # Mostly grass
        elif env_type == 'desert':
            # Red channel for sand, green for dry grass, blue for rock
            splatmap[:, :, 0] = 255  # Mostly sand
        
        return Image.fromarray(splatmap)
    
    async def generate_props(self, concept: Dict[str, Any], mission_type: str) -> List[Dict[str, Any]]:
        """Generate props and interactive elements for the environment"""
        props = []
        env_type = concept['type']
        
        # Add environment-specific props
        if env_type == 'jungle':
            props.extend([
                {'type': 'tree', 'model': 'models/props/jungle/tree_1.fbx', 'density': 0.8},
                {'type': 'rock', 'model': 'models/props/jungle/rock_1.fbx', 'density': 0.3},
                {'type': 'ruins', 'model': 'models/props/jungle/ruins_1.fbx', 'density': 0.1}
            ])
        elif env_type == 'urban':
            props.extend([
                {'type': 'building', 'model': 'models/props/urban/building_1.fbx', 'density': 0.6},
                {'type': 'vehicle', 'model': 'models/props/urban/vehicle_1.fbx', 'density': 0.2},
                {'type': 'barricade', 'model': 'models/props/urban/barricade_1.fbx', 'density': 0.4}
            ])
        
        # Add mission-specific props
        if mission_type == 'hostage_rescue':
            props.append({
                'type': 'hostage',
                'model': 'models/props/characters/hostage.fbx',
                'spawn_points': ['hostage_spawn_1', 'hostage_spawn_2'],
                'interaction': 'rescue'
            })
        elif mission_type == 'destroy_target':
            props.append({
                'type': 'target',
                'model': 'models/props/objectives/target.fbx',
                'spawn_points': ['target_spawn_1'],
                'interaction': 'destroy'
            })
        
        return props
    
    def get_status(self) -> str:
        """Get current agent status"""
        if not self.model_loaded:
            return "offline"
        return "busy" if self.current_task else "idle"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "environment_concept_generation",
            "terrain_generation",
            "texture_creation",
            "prop_placement",
            "lighting_setup"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        torch.cuda.empty_cache()
