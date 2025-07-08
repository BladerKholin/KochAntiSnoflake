import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
import numpy as np
import time

class JAXKochSnowflake:
    def __init__(self):
        # Pre-compile the core functions
        self._koch_step_compiled = jit(self._koch_iteration_step)
    
    @staticmethod
    @jit
    def _koch_transform_segment(p1, p2):
        """Transform a single segment into Koch curve (returns 4 new segments)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate the key points
        p_1_3 = jnp.array([p1[0] + dx/3, p1[1] + dy/3])
        p_2_3 = jnp.array([p1[0] + 2*dx/3, p1[1] + 2*dy/3])
        
        # Calculate the peak point
        mid_x = (p_1_3[0] + p_2_3[0]) / 2
        mid_y = (p_1_3[1] + p_2_3[1]) / 2
        
        segment_length = jnp.sqrt(dx**2 + dy**2)
        height = jnp.sqrt(3) / 6 * segment_length
        
        # Perpendicular vector
        perp_x = -dy / segment_length * height
        perp_y = dx / segment_length * height
        peak = jnp.array([mid_x + perp_x, mid_y + perp_y])
        
        # Return 4 new segments as a 4x2x2 array
        return jnp.array([
            [p1, p_1_3],
            [p_1_3, peak],
            [peak, p_2_3],
            [p_2_3, p2]
        ])
    
    def _koch_iteration_step(self, segments):
        """Apply Koch transformation to all segments vectorized"""
        # Apply transformation to all segments in parallel
        transformed = vmap(self._koch_transform_segment)(segments[:, 0], segments[:, 1])
        
        # Reshape from (N, 4, 2, 2) to (N*4, 2, 2)
        n_segments = segments.shape[0]
        return transformed.reshape(n_segments * 4, 2, 2)
    
    def generate_snowflake_segments(self, iterations=4, size=1.0):
        """Generate Koch snowflake segments using pure JAX operations"""
        # Initial triangle vertices
        height = size * jnp.sqrt(3) / 2
        vertices = jnp.array([
            [-size/2, -height/3],      # Bottom left
            [size/2, -height/3],       # Bottom right  
            [0, 2*height/3]            # Top
        ])
        
        # Initial 3 segments (triangle sides)
        segments = jnp.array([
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]]
        ])
        
        # Apply Koch transformation iteratively
        # Use Python loop with pre-compiled JAX function
        for _ in range(iterations):
            segments = self._koch_step_compiled(segments)
        
        return segments
    
    def segments_to_points(self, segments):
        """Convert segments to ordered points (JIT compiled)"""
        return segments[:, 0]  # Take start point of each segment
    
    def generate_snowflake_points(self, iterations=4, size=1.0):
        """Generate final snowflake points"""
        segments = self.generate_snowflake_segments(iterations, size)
        return self.segments_to_points(segments)

    def save_snowflake(self, filename="koch_jax.png", iterations=4, size=3.0,
                      dpi=300, figsize=(10, 10), line_width=1.0, color='blue'):
        """Generate and save Koch snowflake"""
        
        start_time = time.time()
        print(f"Generating JAX Koch Snowflake with {iterations} iterations...")
        
        # Generate points
        points = self.generate_snowflake_points(iterations, size)
        generation_time = time.time() - start_time
        
        # Convert to numpy for matplotlib
        points_np = np.array(points)
        x_coords = points_np[:, 0]
        y_coords = points_np[:, 1]
        
        # Close the shape
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        
        # Create plot
        plt.figure(figsize=figsize, facecolor='white')
        plt.plot(x_coords, y_coords, color=color, linewidth=line_width)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        
        end_time = time.time()
        
        print(f"Saved as '{filename}'")
        print(f"Points: {len(points)}")
        print(f"Generation time: {generation_time:.4f}s")
        print(f"Total time: {end_time - start_time:.4f}s")
        print("-" * 40)
        
        return {
            'filename': filename,
            'points_count': len(points),
            'generation_time': generation_time,
            'total_time': end_time - start_time
        }
