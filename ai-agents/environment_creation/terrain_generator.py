import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
import noise
import random
from scipy import ndimage

class TerrainGenerator:
    def __init__(self):
        self.terrain_types = {
            'jungle': self.generate_jungle_terrain,
            'desert': self.generate_desert_terrain,
            'arctic': self.generate_arctic_terrain,
            'urban': self.generate_urban_terrain,
            'mountain': self.generate_mountain_terrain
        }
    
    def generate_terrain(self, width: int, height: int, terrain_type: str, 
                        complexity: float = 1.0, roughness: float = 0.5) -> Image.Image:
        """Generate a terrain heightmap based on the specified type"""
        if terrain_type not in self.terrain_types:
            raise ValueError(f"Unknown terrain type: {terrain_type}")
        
        generator = self.terrain_types[terrain_type]
        heightmap = generator(width, height, complexity, roughness)
        
        # Convert to image
        heightmap_img = Image.fromarray((heightmap * 255).astype(np.uint8))
        return heightmap_img
    
    def generate_jungle_terrain(self, width: int, height: int, 
                               complexity: float = 1.0, roughness: float = 0.5) -> np.ndarray:
        """Generate jungle terrain with hills and valleys"""
        scale = 100 * complexity
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        world = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                world[i][j] = noise.pnoise2(i/scale, 
                                          j/scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=1024, 
                                          repeaty=1024, 
                                          base=42)
        
        # Normalize to 0-1 range
        world = (world - world.min()) / (world.max() - world.min())
        
        # Add some random peaks for mountains
        peaks = np.random.rand(height, width) * 0.3
        world = np.clip(world + peaks, 0, 1)
        
        # Apply roughness
        if roughness > 0:
            world = self.apply_roughness(world, roughness)
        
        return world
    
    def generate_desert_terrain(self, width: int, height: int, 
                               complexity: float = 1.0, roughness: float = 0.3) -> np.ndarray:
        """Generate desert terrain with sand dunes"""
        scale = 150 * complexity
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0
        
        world = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                world[i][j] = noise.pnoise2(i/scale, 
                                          j/scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=1024, 
                                          repeaty=1024, 
                                          base=42)
        
        # Normalize to 0-1 range
        world = (world - world.min()) / (world.max() - world.min())
        
        # Create dunes with sine waves
        for i in range(height):
            for j in range(width):
                dune = np.sin(j/20) * 0.2
                world[i][j] = np.clip(world[i][j] + dune, 0, 1)
        
        # Apply roughness
        if roughness > 0:
            world = self.apply_roughness(world, roughness)
        
        return world
    
    def generate_arctic_terrain(self, width: int, height: int, 
                               complexity: float = 1.0, roughness: float = 0.2) -> np.ndarray:
        """Generate arctic terrain with smooth slopes and some mountains"""
        scale = 200 * complexity
        octaves = 8
        persistence = 0.7
        lacunarity = 2.0
        
        world = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                world[i][j] = noise.pnoise2(i/scale, 
                                          j/scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=1024, 
                                          repeaty=1024, 
                                          base=42)
        
        # Normalize to 0-1 range
        world = (world - world.min()) / (world.max() - world.min())
        
        # Make terrain smoother for arctic
        world = ndimage.gaussian_filter(world, sigma=2)
        
        # Add some mountains
        mountains = np.random.rand(height, width) * 0.4
        # Only add mountains to certain areas
        mask = np.random.rand(height, width) > 0.7
        world = np.where(mask, np.clip(world + mountains, 0, 1), world)
        
        # Apply slight roughness
        if roughness > 0:
            world = self.apply_roughness(world, roughness * 0.5)
        
        return world
    
    def generate_urban_terrain(self, width: int, height: int, 
                              complexity: float = 1.0, roughness: float = 0.1) -> np.ndarray:
        """Generate urban terrain - mostly flat with some structures"""
        # Urban areas are mostly flat
        world = np.ones((height, width)) * 0.1
        
        # Add some building foundations
        buildings = np.random.rand(height, width) * 0.3
        # Only add buildings to certain areas
        mask = np.random.rand(height, width) > 0.8
        world = np.where(mask, np.clip(world + buildings, 0, 1), world)
        
        # Add roads (lower areas)
        for i in range(0, height, 20):
            road_width = random.randint(3, 6)
            for j in range(width):
                if j % 50 < road_width:
                    world[i:i+road_width, j] = 0.05
        
        return world
    
    def generate_mountain_terrain(self, width: int, height: int, 
                                 complexity: float = 1.0, roughness: float = 0.8) -> np.ndarray:
        """Generate mountainous terrain with sharp peaks"""
        scale = 80 * complexity
        octaves = 10
        persistence = 0.8
        lacunarity = 2.2
        
        world = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                world[i][j] = noise.pnoise2(i/scale, 
                                          j/scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=1024, 
                                          repeaty=1024, 
                                          base=42)
        
        # Normalize to 0-1 range
        world = (world - world.min()) / (world.max() - world.min())
        
        # Make peaks sharper
        world = np.power(world, 0.7)
        
        # Apply high roughness
        if roughness > 0:
            world = self.apply_roughness(world, roughness)
        
        return world
    
    def apply_roughness(self, heightmap: np.ndarray, roughness: float) -> np.ndarray:
        """Apply roughness to a heightmap"""
        # Generate noise for roughness
        rough_map = np.zeros_like(heightmap)
        for i in range(heightmap.shape[0]):
            for j in range(heightmap.shape[1]):
                rough_map[i][j] = noise.pnoise2(i/30, j/30, octaves=3, base=42)
        
        # Normalize roughness
        rough_map = (rough_map - rough_map.min()) / (rough_map.max() - rough_map.min())
        
        # Apply roughness
        result = heightmap + (rough_map - 0.5) * roughness
        return np.clip(result, 0, 1)
    
    def calculate_slope(self, heightmap: np.ndarray) -> np.ndarray:
        """Calculate slope from heightmap"""
        dx, dy = np.gradient(heightmap)
        slope = np.sqrt(dx**2 + dy**2)
        return slope
    
    def identify_flat_areas(self, heightmap: np.ndarray, slope_threshold: float = 0.05) -> np.ndarray:
        """Identify flat areas in the terrain"""
        slope = self.calculate_slope(heightmap)
        flat_areas = slope < slope_threshold
        return flat_areas
    
    def generate_rivers(self, heightmap: np.ndarray, num_rivers: int = 3) -> np.ndarray:
        """Generate river paths on the terrain"""
        rivers = np.zeros_like(heightmap)
        height, width = heightmap.shape
        
        for _ in range(num_rivers):
            # Start from a high point
            start_x = random.randint(0, width-1)
            start_y = random.randint(0, int(height/4))
            
            x, y = start_x, start_y
            path = []
            
            # Flow downhill
            for _ in range(200):  # Max river length
                if y >= height-1:
                    break
                
                # Mark current position
                path.append((x, y))
                rivers[y, x] = 1.0
                
                # Find the steepest downhill direction
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            neighbors.append((nx, ny, heightmap[ny, nx]))
                
                if not neighbors:
                    break
                
                # Sort by height (lowest first)
                neighbors.sort(key=lambda n: n[2])
                x, y = neighbors[0][0], neighbors[0][1]
            
            # Widen the river
            for x, y in path:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            rivers[ny, nx] = max(rivers[ny, nx], 0.7)
        
        return rivers
    
    def generate_erosion_map(self, heightmap: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Generate an erosion map for the terrain"""
        slope = self.calculate_slope(heightmap)
        erosion = np.zeros_like(heightmap)
        
        # Higher erosion on steeper slopes
        erosion = slope * intensity
        
        # Add some randomness
        random_erosion = np.random.rand(*heightmap.shape) * 0.1
        erosion = np.clip(erosion + random_erosion, 0, 1)
        
        return erosion

# Example usage
if __name__ == "__main__":
    generator = TerrainGenerator()
    
    # Generate a jungle terrain
    terrain = generator.generate_terrain(512, 512, "jungle")
    terrain.save("jungle_terrain.png")
    
    # Generate slope map
    heightmap = np.array(terrain) / 255.0
    slope = generator.calculate_slope(heightmap)
    slope_img = Image.fromarray((slope * 255).astype(np.uint8))
    slope_img.save("slope_map.png")
    
    # Generate rivers
    rivers = generator.generate_rivers(heightmap)
    rivers_img = Image.fromarray((rivers * 255).astype(np.uint8))
    rivers_img.save("rivers.png")
