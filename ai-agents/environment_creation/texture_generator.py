import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Dict, Any, List, Tuple
import random
import noise

class TextureGenerator:
    def __init__(self):
        self.texture_types = {
            'grass': self.generate_grass_texture,
            'dirt': self.generate_dirt_texture,
            'rock': self.generate_rock_texture,
            'sand': self.generate_sand_texture,
            'snow': self.generate_snow_texture,
            'water': self.generate_water_texture,
            'forest': self.generate_forest_texture,
            'urban': self.generate_urban_texture
        }
    
    def generate_texture(self, width: int, height: int, texture_type: str, 
                        color_variation: float = 0.2, scale: float = 1.0) -> Image.Image:
        """Generate a texture of the specified type"""
        if texture_type not in self.texture_types:
            raise ValueError(f"Unknown texture type: {texture_type}")
        
        generator = self.texture_types[texture_type]
        texture = generator(width, height, color_variation, scale)
        return texture
    
    def generate_grass_texture(self, width: int, height: int, 
                              color_variation: float = 0.2, scale: float = 1.0) -> Image.Image:
        """Generate a grass texture"""
        # Base green color
        base_color = (50, 180, 50)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(20*scale), y/(20*scale), octaves=3, base=42)
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 30
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add some texture
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        return img
    
    def generate_dirt_texture(self, width: int, height: int, 
                             color_variation: float = 0.3, scale: float = 1.0) -> Image.Image:
        """Generate a dirt texture"""
        # Base brown color
        base_color = (120, 80, 40)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(15*scale), y/(15*scale), octaves=4, base=42)
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 40
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add some texture
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
        
        return img
    
    def generate_rock_texture(self, width: int, height: int, 
                             color_variation: float = 0.2, scale: float = 1.0) -> Image.Image:
        """Generate a rock texture"""
        # Base gray color
        base_color = (120, 120, 120)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(25*scale), y/(25*scale), octaves=5, base=42)
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 50
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add some sharpness for rock texture
        img = img.filter(ImageFilter.SHARPEN)
        
        return img
    
    def generate_sand_texture(self, width: int, height: int, 
                             color_variation: float = 0.1, scale: float = 1.0) -> Image.Image:
        """Generate a sand texture"""
        # Base sand color
        base_color = (240, 220, 160)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(30*scale), y/(30*scale), octaves=3, base=42)
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 30
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add smoothness for sand
        img = img.filter(ImageFilter.SMOOTH)
        
        return img
    
    def generate_snow_texture(self, width: int, height: int, 
                             color_variation: float = 0.05, scale: float = 1.0) -> Image.Image:
        """Generate a snow texture"""
        # Base white color
        base_color = (250, 250, 250)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add slight variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(40*scale), y/(40*scale), octaves=2, base=42)
                
                # Very slight color variation for snow
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 15
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add slight blur for smooth snow
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return img
    
    def generate_water_texture(self, width: int, height: int, 
                              color_variation: float = 0.1, scale: float = 1.0) -> Image.Image:
        """Generate a water texture"""
        # Base blue color
        base_color = (50, 100, 200)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add wave-like variation
        for y in range(height):
            for x in range(width):
                # Wave pattern using sine waves
                wave1 = np.sin(x/(10*scale) + y/(15*scale)) * 20
                wave2 = np.sin(x/(5*scale) - y/(8*scale)) * 15
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply wave pattern
                r_var += wave1 + wave2
                g_var += wave1 + wave2
                b_var += wave1 + wave2
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add blur for water smoothness
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        return img
    
    def generate_forest_texture(self, width: int, height: int, 
                               color_variation: float = 0.3, scale: float = 1.0) -> Image.Image:
        """Generate a forest floor texture"""
        # Start with dirt texture
        forest = self.generate_dirt_texture(width, height, color_variation, scale)
        pixels = forest.load()
        
        # Add organic matter (leaves, twigs)
        for y in range(height):
            for x in range(width):
                # Random organic spots
                if random.random() < 0.02:
                    # Dark brown for organic matter
                    pixels[x, y] = (60, 40, 20)
                
                # Random green spots for moss
                if random.random() < 0.01:
                    pixels[x, y] = (70, 120, 70)
        
        return forest
    
    def generate_urban_texture(self, width: int, height: int, 
                              color_variation: float = 0.2, scale: float = 1.0) -> Image.Image:
        """Generate an urban texture (concrete, asphalt)"""
        # Base gray color
        base_color = (100, 100, 100)
        
        # Create base image
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        # Add variation
        for y in range(height):
            for x in range(width):
                # Perlin noise for natural variation
                n = noise.pnoise2(x/(25*scale), y/(25*scale), octaves=4, base=42)
                
                # Color variation
                r_var = random.randint(-int(255*color_variation), int(255*color_variation))
                g_var = random.randint(-int(255*color_variation), int(255*color_variation))
                b_var = random.randint(-int(255*color_variation), int(255*color_variation))
                
                # Apply noise-based variation
                noise_factor = n * 40
                r_var += noise_factor
                g_var += noise_factor
                b_var += noise_factor
                
                # Get original pixel
                r, g, b = pixels[x, y]
                
                # Apply variation
                r = int(np.clip(r + r_var, 0, 255))
                g = int(np.clip(g + g_var, 0, 255))
                b = int(np.clip(b + b_var, 0, 255))
                
                pixels[x, y] = (r, g, b)
        
        # Add grid pattern for urban areas
        for y in range(0, height, 20):
            for x in range(0, width, 20):
                # Darker grid lines
                for i in range(2):
                    for j in range(width):
                        if y+i < height:
                            r, g, b = pixels[x+j, y+i]
                            pixels[x+j, y+i] = (r//2, g//2, b//2)
                
                for i in range(2):
                    for j in range(height):
                        if x+i < width:
                            r, g, b = pixels[x+i, y+j]
                            pixels[x+i, y+j] = (r//2, g//2, b//2)
        
        return img
    
    def generate_normal_map(self, heightmap: Image.Image, strength: float = 1.0) -> Image.Image:
        """Generate a normal map from a heightmap"""
        # Convert to numpy array
        height_array = np.array(heightmap.convert('L'), dtype=np.float32) / 255.0
        
        # Calculate gradients
        dx, dy = np.gradient(height_array)
        
        # Calculate normal components
        normal = np.dstack((-dx * strength, -dy * strength, np.ones_like(height_array)))
        norm = np.sqrt(np.sum(normal**2, axis=2))
        normal[:, :, 0] /= norm
        normal[:, :, 1] /= norm
        normal[:, :, 2] /= norm
        
        # Convert to 0-255 range
        normal = (normal + 1) * 0.5 * 255
        normal = normal.astype(np.uint8)
        
        return Image.fromarray(normal, 'RGB')
    
    def generate_specular_map(self, texture: Image.Image, metallic_factor: float = 0.5) -> Image.Image:
        """Generate a specular map from a texture"""
        # Convert to grayscale
        gray = texture.convert('L')
        pixels = gray.load()
        
        # Adjust based on metallic factor
        specular = Image.new('L', texture.size)
        spec_pixels = specular.load()
        
        for y in range(texture.size[1]):
            for x in range(texture.size[0]):
                # Darker areas are less specular
                value = pixels[x, y]
                # Adjust based on metallic factor
                spec_value = int(value * metallic_factor)
                spec_pixels[x, y] = spec_value
        
        return specular
    
    def generate_ao_map(self, heightmap: Image.Image, intensity: float = 1.0) -> Image.Image:
        """Generate an ambient occlusion map from a heightmap"""
        # Convert to numpy array
        height_array = np.array(heightmap.convert('L'), dtype=np.float32) / 255.0
        
        # Create AO map
        ao_map = np.ones_like(height_array)
        height, width = height_array.shape
        
        # Simple AO calculation
        for y in range(1, height-1):
            for x in range(1, width-1):
                # Sample surrounding heights
                surroundings = [
                    height_array[y-1, x-1], height_array[y-1, x], height_array[y-1, x+1],
                    height_array[y, x-1], height_array[y, x+1],
                    height_array[y+1, x-1], height_array[y+1, x], height_array[y+1, x+1]
                ]
                
                # Calculate AO based on height differences
                current = height_array[y, x]
                occlusion = 0
                for h in surroundings:
                    if h > current:
                        occlusion += (h - current) * 0.5
                
                # Apply occlusion
                ao_map[y, x] = 1.0 - min(occlusion * intensity, 0.5)
        
        # Convert to image
        ao_map = (ao_map * 255).astype(np.uint8)
        return Image.fromarray(ao_map, 'L')

# Example usage
if __name__ == "__main__":
    generator = TextureGenerator()
    
    # Generate different textures
    grass = generator.generate_texture(512, 512, "grass")
    grass.save("grass_texture.png")
    
    rock = generator.generate_texture(512, 512, "rock")
    rock.save("rock_texture.png")
    
    # Generate normal map
    normal_map = generator.generate_normal_map(grass)
    normal_map.save("grass_normal.png")
    
    # Generate specular map
    specular_map = generator.generate_specular_map(grass)
    specular_map.save("grass_specular.png")
