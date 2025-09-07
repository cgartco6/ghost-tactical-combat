import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random

class EnvironmentCreationAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.environment_db = {}
        self.logger = self.setup_logging()
        
        # Initialize models
        self.setup_models()
    
    def setup_logging(self):
        """Setup logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/environment_agent.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('EnvironmentCreationAgent')
    
    def setup_models(self):
        """Initialize AI models for environment creation"""
        try:
            self.logger.info("Loading environment creation models...")
            
            # Load Stable Diffusion for environment generation
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
            
            # Load inpainting model for environment editing
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
            # Load language model for environment description
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.lm_model.to(self.device)
            
            self.model_loaded = True
            self.logger.info("Environment creation models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info("Initializing EnvironmentCreationAgent...")
        
        # Load existing environments from database
        await self.load_environment_database()
        
        self.logger.info("EnvironmentCreationAgent initialized successfully")
    
    async def load_environment_database(self):
        """Load environment database from file"""
        try:
            db_path = Path("data/environments/environment_db.json")
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self.environment_db = json.load(f)
                self.logger.info(f"Loaded {len(self.environment_db)} environments from database")
            else:
                self.logger.info("No existing environment database found")
        except Exception as e:
            self.logger.error(f"Error loading environment database: {str(e)}")
    
    async def save_environment_database(self):
        """Save environment database to file"""
        try:
            db_path = Path("data/environments/environment_db.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(self.environment_db, f, indent=2)
            
            self.logger.info("Environment database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving environment database: {str(e)}")
    
    async def generate_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new environment with the given parameters"""
        self.current_task = "generate_environment"
        
        try:
            # Generate environment concept
            concept = await self.generate_environment_concept(params)
            
            # Generate visual assets
            assets = await self.generate_environment_assets(concept)
            
            # Generate terrain data
            terrain = await self.generate_terrain(concept)
            
            # Generate props and interactive elements
            props = await self.generate_props(concept)
            
            # Generate lighting and weather settings
            lighting = await self.generate_lighting_settings(concept)
            
            # Generate navigation mesh
            navmesh = await self.generate_navmesh(concept, terrain)
            
            # Compile environment data
            environment_id = self.generate_environment_id(concept)
            environment_data = {
                'id': environment_id,
                'concept': concept,
                'assets': assets,
                'terrain': terrain,
                'props': props,
                'lighting': lighting,
                'navmesh': navmesh,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save to database
            self.environment_db[environment_id] = environment_data
            await self.save_environment_database()
            
            # Export to game format
            await self.export_environment(environment_data)
            
            return {
                'success': True,
                'environment_id': environment_id,
                'environment_data': environment_data
            }
            
        except Exception as e:
            self.logger.error(f"Error creating environment: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_environment_concept(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate environment concept based on parameters"""
        # Extract parameters
        environment_type = params.get('type', 'jungle')
        biome = params.get('biome', 'tropical')
        time_of_day = params.get('time_of_day', 'day')
        weather = params.get('weather', 'clear')
        mission_type = params.get('mission_type', 'infiltration')
        
        # Generate concept using AI
        prompt = self.create_environment_prompt(environment_type, biome, time_of_day, weather, mission_type)
        
        # Generate description using language model
        description = await self.generate_environment_description(prompt)
        
        return {
            'type': environment_type,
            'biome': biome,
            'time_of_day': time_of_day,
            'weather': weather,
            'mission_type': mission_type,
            'prompt': prompt,
            'description': description,
            'size': self.generate_environment_size(mission_type),
            'complexity': self.generate_environment_complexity(mission_type)
        }
    
    def create_environment_prompt(self, env_type: str, biome: str, time_of_day: str, weather: str, mission_type: str) -> str:
        """Create a prompt for environment generation"""
        prompts = {
            'jungle': f"Lush {biome} jungle environment, dense foliage, ancient ruins, {time_of_day} time, {weather} weather, tactical combat game environment, 4k, photorealistic, unreal engine 5",
            'urban': f"War-torn urban environment, destroyed buildings, rubble, {time_of_day} time, {weather} weather, tactical combat game environment, 4k, photorealistic, unreal engine 5",
            'desert': f"Arid {biome} desert environment, sand dunes, rocky outcrops, {time_of_day} time, {weather} weather, tactical combat game environment, 4k, photorealistic, unreal engine 5",
            'arctic': f"Snow-covered arctic environment, icy mountains, frozen lakes, {time_of_day} time, {weather} weather, tactical combat game environment, 4k, photorealistic, unreal engine 5",
            'facility': f"Secret military facility, high-tech equipment, industrial structures, {time_of_day} time, {weather} weather, tactical combat game environment, 4k, photorealistic, unreal engine 5"
        }
        
        return prompts.get(env_type, prompts['jungle'])
    
    async def generate_environment_description(self, prompt: str) -> str:
        """Generate environment description using language model"""
        description_prompt = f"Describe this game environment in detail: {prompt}"
        
        inputs = self.tokenizer.encode(description_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=150, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_environment_size(self, mission_type: str) -> Dict[str, int]:
        """Generate appropriate environment size based on mission type"""
        sizes = {
            'infiltration': {'width': 256, 'height': 256, 'complexity': 'high'},
            'assault': {'width': 512, 'height': 512, 'complexity': 'medium'},
            'defense': {'width': 384, 'height': 384, 'complexity': 'high'},
            'extraction': {'width': 512, 'height': 256, 'complexity': 'medium'},
            'recon': {'width': 1024, 'height': 1024, 'complexity': 'low'}
        }
        
        return sizes.get(mission_type, sizes['infiltration'])
    
    def generate_environment_complexity(self, mission_type: str) -> str:
        """Generate environment complexity based on mission type"""
        complexities = {
            'infiltration': 'high',
            'assault': 'medium',
            'defense': 'high',
            'extraction': 'medium',
            'recon': 'low'
        }
        
        return complexities.get(mission_type, 'medium')
    
    async def generate_environment_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual assets for the environment"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        assets = {}
        env_type = concept['type']
        env_id = self.generate_environment_id(concept)
        
        # Generate terrain texture
        terrain_texture = await self.generate_terrain_texture(concept)
        assets['terrain_texture'] = terrain_texture
        
        # Generate skybox
        skybox = await self.generate_skybox(concept)
        assets['skybox'] = skybox
        
        # Generate environment-specific assets
        if env_type == 'jungle':
            assets['foliage'] = await self.generate_jungle_assets(concept)
        elif env_type == 'urban':
            assets['buildings'] = await self.generate_urban_assets(concept)
        elif env_type == 'desert':
            assets['rock_formations'] = await self.generate_desert_assets(concept)
        elif env_type == 'arctic':
            assets['ice_formations'] = await self.generate_arctic_assets(concept)
        elif env_type == 'facility':
            assets['structures'] = await self.generate_facility_assets(concept)
        
        return assets
    
    async def generate_terrain_texture(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate terrain texture for the environment"""
        prompt = f"{concept['prompt']}, terrain texture, seamless, tileable"
        
        with torch.no_grad():
            image = self.sd_pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]
        
        # Save texture
        env_id = self.generate_environment_id(concept)
        texture_path = f"assets/environments/{env_id}/textures/terrain_diffuse.png"
        
        Path(texture_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(texture_path)
        
        # Generate normal map (placeholder - would use actual normal map generation)
        normal_path = texture_path.replace('diffuse', 'normal')
        self.generate_normal_map(texture_path, normal_path)
        
        return {
            'diffuse': texture_path,
            'normal': normal_path,
            'roughness': self.generate_roughness_map(concept, texture_path),
            'ao': self.generate_ao_map(concept, texture_path)
        }
    
    def generate_normal_map(self, diffuse_path: str, normal_path: str):
        """Generate a normal map from a diffuse texture (placeholder implementation)"""
        # In a real implementation, this would use proper normal map generation
        # For now, we'll create a placeholder normal map
        diffuse_img = Image.open(diffuse_path)
        width, height = diffuse_img.size
        
        # Create a simple normal map (blue with some variation)
        normal_img = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # Simple normal map (mostly blue with some red/green for variation)
                r = random.randint(120, 140)
                g = random.randint(120, 140)
                b = random.randint(240, 255)
                pixels.append((r, g, b))
        
        normal_img.putdata(pixels)
        normal_img.save(normal_path)
    
    def generate_roughness_map(self, concept: Dict[str, Any], texture_path: str) -> str:
        """Generate a roughness map for the terrain"""
        env_id = self.generate_environment_id(concept)
        roughness_path = f"assets/environments/{env_id}/textures/terrain_roughness.png"
        
        # Create roughness map based on environment type
        roughness_values = {
            'jungle': 0.7,
            'urban': 0.5,
            'desert': 0.9,
            'arctic': 0.3,
            'facility': 0.4
        }
        
        roughness = roughness_values.get(concept['type'], 0.5)
        
        # Create a simple roughness map
        img = Image.open(texture_path)
        width, height = img.size
        
        roughness_img = Image.new('L', (width, height))
        pixels = []
        
        for _ in range(width * height):
            # Add some variation to the roughness
            var = random.uniform(-0.1, 0.1)
            value = int(min(max(roughness + var, 0), 1) * 255)
            pixels.append(value)
        
        roughness_img.putdata(pixels)
        roughness_img.save(roughness_path)
        
        return roughness_path
    
    def generate_ao_map(self, concept: Dict[str, Any], texture_path: str) -> str:
        """Generate an ambient occlusion map for the terrain"""
        env_id = self.generate_environment_id(concept)
        ao_path = f"assets/environments/{env_id}/textures/terrain_ao.png"
        
        # Create a simple AO map
        img = Image.open(texture_path)
        width, height = img.size
        
        ao_img = Image.new('L', (width, height))
        pixels = []
        
        for _ in range(width * height):
            # Random AO values with some pattern
            value = random.randint(200, 255)
            pixels.append(value)
        
        ao_img.putdata(pixels)
        ao_img.save(ao_path)
        
        return ao_path
    
    async def generate_skybox(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate skybox for the environment"""
        prompt = f"Skybox for {concept['prompt']}, 360 panorama, HDR, realistic"
        
        with torch.no_grad():
            image = self.sd_pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=2048,
                height=1024
            ).images[0]
        
        # Save skybox
        env_id = self.generate_environment_id(concept)
        skybox_path = f"assets/environments/{env_id}/skybox/skybox.png"
        
        Path(skybox_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(skybox_path)
        
        return {
            'texture': skybox_path,
            'type': 'panoramic',
            'light_intensity': self.calculate_light_intensity(concept)
        }
    
    def calculate_light_intensity(self, concept: Dict[str, Any]) -> float:
        """Calculate light intensity based on time of day and weather"""
        time_intensity = {
            'dawn': 0.6,
            'day': 1.0,
            'dusk': 0.7,
            'night': 0.2
        }
        
        weather_intensity = {
            'clear': 1.0,
            'cloudy': 0.7,
            'rain': 0.5,
            'fog': 0.4,
            'storm': 0.3
        }
        
        time = concept.get('time_of_day', 'day')
        weather = concept.get('weather', 'clear')
        
        return time_intensity.get(time, 1.0) * weather_intensity.get(weather, 1.0)
    
    async def generate_jungle_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate jungle-specific assets"""
        assets = {}
        env_id = self.generate_environment_id(concept)
        
        # Generate foliage assets
        foliage_types = ['tree', 'bush', 'fern', 'vines', 'grass']
        
        for foliage in foliage_types:
            prompt = f"{foliage} for {concept['prompt']}, 4k, photorealistic, game asset"
            
            with torch.no_grad():
                image = self.sd_pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
            
            asset_path = f"assets/environments/{env_id}/foliage/{foliage}.png"
            Path(asset_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(asset_path)
            
            assets[foliage] = {
                'texture': asset_path,
                'model': f"models/foliage/{foliage}.fbx",
                'density': random.uniform(0.1, 0.8)
            }
        
        return assets
    
    async def generate_urban_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate urban-specific assets"""
        assets = {}
        env_id = self.generate_environment_id(concept)
        
        # Generate building assets
        building_types = ['building_ruined', 'building_intact', 'wall', 'barricade', 'vehicle']
        
        for building in building_types:
            prompt = f"{building} for {concept['prompt']}, 4k, photorealistic, game asset"
            
            with torch.no_grad():
                image = self.sd_pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
            
            asset_path = f"assets/environments/{env_id}/buildings/{building}.png"
            Path(asset_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(asset_path)
            
            assets[building] = {
                'texture': asset_path,
                'model': f"models/buildings/{building}.fbx",
                'destructible': random.random() > 0.5
            }
        
        return assets
    
    async def generate_terrain(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate terrain data for the environment"""
        env_id = self.generate_environment_id(concept)
        
        # Generate heightmap
        heightmap = self.generate_heightmap(concept)
        heightmap_path = f"assets/environments/{env_id}/terrain/heightmap.png"
        Path(heightmap_path).parent.mkdir(parents=True, exist_ok=True)
        heightmap.save(heightmap_path)
        
        # Generate splatmap for texture blending
        splatmap = self.generate_splatmap(concept)
        splatmap_path = f"assets/environments/{env_id}/terrain/splatmap.png"
        splatmap.save(splatmap_path)
        
        return {
            'heightmap': heightmap_path,
            'splatmap': splatmap_path,
            'size': concept['size'],
            'max_height': self.calculate_max_height(concept),
            'texture_blend': self.generate_texture_blend_info(concept)
        }
    
    def generate_heightmap(self, concept: Dict[str, Any]) -> Image.Image:
        """Generate a heightmap for the terrain"""
        width, height = concept['size']['width'], concept['size']['height']
        heightmap = Image.new('L', (width, height))
        pixels = []
        
        # Generate different height patterns based on environment type
        env_type = concept['type']
        
        if env_type == 'jungle':
            # Jungle has varied terrain with hills and valleys
            for y in range(height):
                for x in range(width):
                    # Generate hilly terrain
                    value = int(self.generate_jungle_height(x, y, width, height) * 255)
                    pixels.append(value)
        
        elif env_type == 'desert':
            # Desert has sand dunes
            for y in range(height):
                for x in range(width):
                    value = int(self.generate_desert_height(x, y, width, height) * 255)
                    pixels.append(value)
        
        elif env_type == 'arctic':
            # Arctic has smooth terrain with some mountains
            for y in range(height):
                for x in range(width):
                    value = int(self.generate_arctic_height(x, y, width, height) * 255)
                    pixels.append(value)
        
        else:
            # Default terrain (mostly flat with some variation)
            for y in range(height):
                for x in range(width):
                    value = int(self.generate_default_height(x, y, width, height) * 255)
                    pixels.append(value)
        
        heightmap.putdata(pixels)
        return heightmap
    
    def generate_jungle_height(self, x: int, y: int, width: int, height: int) -> float:
        """Generate height value for jungle terrain"""
        # Simplex noise would be better here, but for simplicity we'll use sine waves
        x_norm = x / width
        y_norm = y / height
        
        # Create hills and valleys
        value = (np.sin(x_norm * 10) + np.cos(y_norm * 8)) / 2  # Range -1 to 1
        value = (value + 1) / 2  # Range 0 to 1
        
        # Add some randomness
        value += random.uniform(-0.1, 0.1)
        
        return max(0, min(1, value))
    
    def generate_desert_height(self, x: int, y: int, width: int, height: int) -> float:
        """Generate height value for desert terrain"""
        x_norm = x / width
        y_norm = y / height
        
        # Create sand dunes
        value = np.sin(x_norm * 5) * np.cos(y_norm * 3)  # Range -1 to 1
        value = (value + 1) / 2  # Range 0 to 1
        
        # Flatten some areas
        if random.random() < 0.3:
            value *= 0.3
        
        return max(0, min(1, value))
    
    def generate_arctic_height(self, x: int, y: int, width: int, height: int) -> float:
        """Generate height value for arctic terrain"""
        x_norm = x / width
        y_norm = y / height
        
        # Mostly flat with some mountains
        value = 0.2  # Base height
        
        # Add mountains in some areas
        if (x_norm > 0.7 and x_norm < 0.9) or (y_norm > 0.7 and y_norm < 0.9):
            mountain = np.sin(x_norm * 20) * np.cos(y_norm * 15)
            value += max(0, mountain) * 0.5
        
        return max(0, min(1, value))
    
    def generate_default_height(self, x: int, y: int, width: int, height: int) -> float:
        """Generate height value for default terrain"""
        # Mostly flat with slight variation
        return 0.3 + random.uniform(-0.05, 0.05)
    
    def generate_splatmap(self, concept: Dict[str, Any]) -> Image.Image:
        """Generate a splatmap for texture blending"""
        width, height = concept['size']['width'], concept['size']['height']
        splatmap = Image.new('RGB', (width, height))
        pixels = []
        
        env_type = concept['type']
        
        for y in range(height):
            for x in range(width):
                if env_type == 'jungle':
                    # Green channel for grass, red for dirt, blue for rock
                    r = random.randint(0, 50)  # Some dirt
                    g = random.randint(200, 255)  # Mostly grass
                    b = random.randint(0, 100)  # Some rock
                    pixels.append((r, g, b))
                
                elif env_type == 'desert':
                    # Red channel for sand, green for dry grass, blue for rock
                    r = random.randint(200, 255)  # Mostly sand
                    g = random.randint(50, 150)  # Some dry grass
                    b = random.randint(0, 100)  # Some rock
                    pixels.append((r, g, b))
                
                elif env_type == 'arctic':
                    # White for snow, blue for ice
                    r = random.randint(200, 255)
                    g = random.randint(200, 255)
                    b = random.randint(200, 255)
                    pixels.append((r, g, b))
                
                else:
                    # Default: gray for urban areas
                    r = random.randint(100, 150)
                    g = random.randint(100, 150)
                    b = random.randint(100, 150)
                    pixels.append((r, g, b))
        
        splatmap.putdata(pixels)
        return splatmap
    
    def calculate_max_height(self, concept: Dict[str, Any]) -> float:
        """Calculate maximum terrain height based on environment type"""
        heights = {
            'jungle': 256,  # Varied terrain with hills
            'urban': 50,    # Mostly flat with buildings
            'desert': 100,  # Sand dunes
            'arctic': 300,  # Mountains
            'facility': 20  # Very flat
        }
        
        return heights.get(concept['type'], 100)
    
    def generate_texture_blend_info(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate texture blending information"""
        env_type = concept['type']
        
        if env_type == 'jungle':
            return {
                'textures': [
                    {'path': 'textures/terrain/grass.jpg', 'channel': 'g'},
                    {'path': 'textures/terrain/dirt.jpg', 'channel': 'r'},
                    {'path': 'textures/terrain/rock.jpg', 'channel': 'b'}
                ],
                'tiling': [10, 10, 5]
            }
        
        elif env_type == 'desert':
            return {
                'textures': [
                    {'path': 'textures/terrain/sand.jpg', 'channel': 'r'},
                    {'path': 'textures/terrain/dry_grass.jpg', 'channel': 'g'},
                    {'path': 'textures/terrain/rock.jpg', 'channel': 'b'}
                ],
                'tiling': [15, 15, 5]
            }
        
        else:
            return {
                'textures': [
                    {'path': 'textures/terrain/concrete.jpg', 'channel': 'r'},
                    {'path': 'textures/terrain/asphalt.jpg', 'channel': 'g'},
                    {'path': 'textures/terrain/rubble.jpg', 'channel': 'b'}
                ],
                'tiling': [8, 8, 8]
            }
    
    async def generate_props(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate props and interactive elements for the environment"""
        props = []
        env_type = concept['type']
        mission_type = concept['mission_type']
        
        # Add environment-specific props
        if env_type == 'jungle':
            props.extend([
                {
                    'type': 'tree',
                    'model': 'models/props/jungle/tree_1.fbx',
                    'texture': 'textures/props/tree_1.jpg',
                    'density': 0.7,
                    'collision': True,
                    'destructible': False
                },
                {
                    'type': 'rock',
                    'model': 'models/props/jungle/rock_1.fbx',
                    'texture': 'textures/props/rock_1.jpg',
                    'density': 0.3,
                    'collision': True,
                    'destructible': False
                },
                {
                    'type': 'ruins',
                    'model': 'models/props/jungle/ruins_1.fbx',
                    'texture': 'textures/props/ruins_1.jpg',
                    'density': 0.1,
                    'collision': True,
                    'destructible': True
                }
            ])
        
        elif env_type == 'urban':
            props.extend([
                {
                    'type': 'building',
                    'model': 'models/props/urban/building_1.fbx',
                    'texture': 'textures/props/building_1.jpg',
                    'density': 0.6,
                    'collision': True,
                    'destructible': True
                },
                {
                    'type': 'vehicle',
                    'model': 'models/props/urban/vehicle_1.fbx',
                    'texture': 'textures/props/vehicle_1.jpg',
                    'density': 0.2,
                    'collision': True,
                    'destructible': True
                },
                {
                    'type': 'barricade',
                    'model': 'models/props/urban/barricade_1.fbx',
                    'texture': 'textures/props/barricade_1.jpg',
                    'density': 0.4,
                    'collision': True,
                    'destructible': True
                }
            ])
        
        # Add mission-specific props
        if mission_type == 'hostage_rescue':
            props.append({
                'type': 'hostage',
                'model': 'models/props/characters/hostage.fbx',
                'texture': 'textures/characters/hostage.jpg',
                'spawn_points': ['hostage_spawn_1', 'hostage_spawn_2'],
                'interaction': 'rescue',
                'collision': True,
                'destructible': False
            })
        
        elif mission_type == 'destroy_target':
            props.append({
                'type': 'target',
                'model': 'models/props/objectives/target.fbx',
                'texture': 'textures/props/target.jpg',
                'spawn_points': ['target_spawn_1'],
                'interaction': 'destroy',
                'collision': True,
                'destructible': True
            })
        
        elif mission_type == 'extraction':
            props.append({
                'type': 'extraction_point',
                'model': 'models/props/objectives/extraction.fbx',
                'texture': 'textures/props/extraction.jpg',
                'spawn_points': ['extraction_spawn'],
                'interaction': 'extract',
                'collision': False,
                'destructible': False
            })
        
        return props
    
    async def generate_lighting_settings(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lighting settings for the environment"""
        time_of_day = concept['time_of_day']
        weather = concept['weather']
        
        # Base lighting settings
        lighting = {
            'ambient_intensity': 0.4,
            'sun_intensity': 1.0,
            'sun_color': '#ffffff',
            'ambient_color': '#ffffff',
            'fog_enabled': True,
            'fog_density': 0.01,
            'fog_color': '#ffffff',
            'reflections_enabled': True,
            'shadows_enabled': True
        }
        
        # Adjust based on time of day
        if time_of_day == 'dawn':
            lighting['sun_intensity'] = 0.7
            lighting['sun_color'] = '#ff7b38'
            lighting['ambient_color'] = '#383c5c'
            lighting['fog_color'] = '#5c7b9c'
        
        elif time_of_day == 'day':
            lighting['sun_intensity'] = 1.0
            lighting['sun_color'] = '#ffffff'
            lighting['ambient_color'] = '#a4c5e8'
            lighting['fog_color'] = '#a4c5e8'
        
        elif time_of_day == 'dusk':
            lighting['sun_intensity'] = 0.6
            lighting['sun_color'] = '#ff6b47'
            lighting['ambient_color'] = '#47385c'
            lighting['fog_color'] = '#9c7b8c'
        
        elif time_of_day == 'night':
            lighting['sun_intensity'] = 0.1
            lighting['sun_color'] = '#1e3a5f'
            lighting['ambient_color'] = '#0c1b33'
            lighting['ambient_intensity'] = 0.2
            lighting['fog_color'] = '#0c1b33'
            lighting['fog_density'] = 0.05
        
        # Adjust based on weather
        if weather == 'cloudy':
            lighting['sun_intensity'] *= 0.6
            lighting['ambient_intensity'] *= 0.8
        
        elif weather == 'rain':
            lighting['sun_intensity'] *= 0.4
            lighting['ambient_intensity'] *= 0.7
            lighting['fog_density'] = 0.03
        
        elif weather == 'fog':
            lighting['sun_intensity'] *= 0.5
            lighting['ambient_intensity'] *= 0.6
            lighting['fog_density'] = 0.05
        
        elif weather == 'storm':
            lighting['sun_intensity'] *= 0.3
            lighting['ambient_intensity'] *= 0.5
            lighting['fog_density'] = 0.04
        
        return lighting
    
    async def generate_navmesh(self, concept: Dict[str, Any], terrain: Dict[str, Any]) -> Dict[str, Any]:
        """Generate navigation mesh for the environment"""
        env_id = self.generate_environment_id(concept)
        
        # Generate navmesh data (simplified)
        navmesh_data = {
            'walkable_areas': await self.calculate_walkable_areas(terrain),
            'jump_links': await self.generate_jump_links(concept),
            'cover_points': await self.generate_cover_points(concept),
            'path_nodes': await self.generate_path_nodes(concept)
        }
        
        # Save navmesh
        navmesh_path = f"assets/environments/{env_id}/navigation/navmesh.json"
        Path(navmesh_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(navmesh_path, 'w') as f:
            json.dump(navmesh_data, f, indent=2)
        
        return {
            'data': navmesh_path,
            'generated_at': datetime.now().isoformat()
        }
    
    async def calculate_walkable_areas(self, terrain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate walkable areas based on terrain"""
        # Simplified implementation
        # In a real scenario, this would analyze the heightmap and slope
        return [
            {
                'id': 'area_1',
                'bounds': {'min_x': 0, 'max_x': 256, 'min_y': 0, 'max_y': 256},
                'type': 'walkable',
                'slope': 15.0
            }
        ]
    
    async def generate_jump_links(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate jump links for navigation"""
        # Simplified implementation
        return [
            {
                'from': {'x': 50, 'y': 50, 'z': 10},
                'to': {'x': 60, 'y': 60, 'z': 10},
                'type': 'jump',
                'distance': 15.0
            }
        ]
    
    async def generate_cover_points(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cover points for tactical positioning"""
        # Simplified implementation
        cover_points = []
        
        for i in range(20):
            cover_points.append({
                'id': f'cover_{i}',
                'position': {
                    'x': random.randint(0, concept['size']['width']),
                    'y': random.randint(0, concept['size']['height']),
                    'z': 0
                },
                'type': random.choice(['low', 'high', 'left', 'right']),
                'quality': random.uniform(0.5, 1.0)
            })
        
        return cover_points
    
    async def generate_path_nodes(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate path nodes for AI navigation"""
        # Simplified implementation
        path_nodes = []
        grid_size = 32
        
        for x in range(0, concept['size']['width'], grid_size):
            for y in range(0, concept['size']['height'], grid_size):
                path_nodes.append({
                    'id': f'node_{x}_{y}',
                    'position': {'x': x, 'y': y, 'z': 0},
                    'connections': self.generate_node_connections(x, y, grid_size, concept['size'])
                })
        
        return path_nodes
    
    def generate_node_connections(self, x: int, y: int, grid_size: int, size: Dict[str, int]) -> List[str]:
        """Generate connections between path nodes"""
        connections = []
        
        # Check all 8 directions
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx * grid_size, y + dy * grid_size
            
            # Check if within bounds
            if 0 <= nx < size['width'] and 0 <= ny < size['height']:
                connections.append(f'node_{nx}_{ny}')
        
        return connections
    
    def generate_environment_id(self, concept: Dict[str, Any]) -> str:
        """Generate a unique ID for the environment"""
        timestamp = int(datetime.now().timestamp())
        type_abbr = concept['type'][:3].upper()
        biome_abbr = concept['biome'][:3].upper()
        random_suffix = np.random.randint(1000, 9999)
        
        return f"ENV_{type_abbr}_{biome_abbr}_{timestamp}_{random_suffix}"
    
    async def export_environment(self, environment_data: Dict[str, Any]):
        """Export environment data to game format"""
        try:
            env_id = environment_data['id']
            export_path = f"exports/environments/{env_id}"
            
            # Create export directory
            Path(export_path).mkdir(parents=True, exist_ok=True)
            
            # Export environment data as JSON
            with open(f"{export_path}/environment.json", 'w') as f:
                json.dump(environment_data, f, indent=2)
            
            # Export for Unity
            await self.export_unity_environment(environment_data, export_path)
            
            # Export for Unreal Engine
            await self.export_unreal_environment(environment_data, export_path)
            
            self.logger.info(f"Exported environment {env_id} to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting environment: {str(e)}")
    
    async def export_unity_environment(self, environment_data: Dict[str, Any], export_path: str):
        """Export environment data for Unity game engine"""
        unity_data = {
            'name': environment_data['id'],
            'prefabPath': f"Environments/{environment_data['id']}/Prefabs/Environment.prefab",
            'terrain': {
                'heightmap': environment_data['terrain']['heightmap'],
                'splatmap': environment_data['terrain']['splatmap'],
                'size': environment_data['terrain']['size'],
                'maxHeight': environment_data['terrain']['max_height']
            },
            'lighting': environment_data['lighting'],
            'props': environment_data['props'],
            'navmesh': environment_data['navmesh']['data']
        }
        
        unity_path = f"{export_path}/unity"
        Path(unity_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unity_path}/environment_data.json", 'w') as f:
            json.dump(unity_data, f, indent=2)
    
    async def export_unreal_environment(self, environment_data: Dict[str, Any], export_path: str):
        """Export environment data for Unreal Engine"""
        unreal_data = {
            'name': environment_data['id'],
            'blueprintPath': f"/Game/Environments/{environment_data['id']}/BP_Environment.BP_Environment",
            'terrain': {
                'heightmap': environment_data['terrain']['heightmap'],
                'splatmap': environment_data['terrain']['splatmap'],
                'size': environment_data['terrain']['size'],
                'maxHeight': environment_data['terrain']['max_height']
            },
            'lighting': environment_data['lighting'],
            'foliage': environment_data['assets'].get('foliage', {}),
            'props': environment_data['props'],
            'navmesh': environment_data['navmesh']['data']
        }
        
        unreal_path = f"{export_path}/unreal"
        Path(unreal_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unreal_path}/environment_data.json", 'w') as f:
            json.dump(unreal_data, f, indent=2)
    
    async def update_environment(self, environment_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing environment"""
        if environment_id not in self.environment_db:
            return {'success': False, 'error': 'Environment not found'}
        
        try:
            # Update environment data
            self.environment_db[environment_id].update(updates)
            self.environment_db[environment_id]['updated_at'] = datetime.now().isoformat()
            self.environment_db[environment_id]['version'] = str(float(self.environment_db[environment_id]['version']) + 0.1)
            
            # Save to database
            await self.save_environment_database()
            
            # Re-export environment
            await self.export_environment(self.environment_db[environment_id])
            
            return {
                'success': True,
                'environment_id': environment_id,
                'updated_data': self.environment_db[environment_id]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_environment(self, environment_id: str) -> Optional[Dict[str, Any]]:
        """Get environment data by ID"""
        return self.environment_db.get(environment_id)
    
    async def list_environments(self, filter_type: str = None) -> List[Dict[str, Any]]:
        """List all environments, optionally filtered by type"""
        environments = list(self.environment_db.values())
        
        if filter_type:
            environments = [e for e in environments if e['concept']['type'] == filter_type]
        
        return environments
    
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
            "lighting_setup",
            "navmesh_generation",
            "environment_export"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Shutting down EnvironmentCreationAgent...")
        
        # Clear model resources
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        if hasattr(self, 'inpaint_pipeline'):
            del self.inpaint_pipeline
        if hasattr(self, 'lm_model'):
            del self.lm_model
        
        torch.cuda.empty_cache()
        self.logger.info("EnvironmentCreationAgent shutdown complete")
