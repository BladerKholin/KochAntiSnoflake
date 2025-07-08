import matplotlib.pyplot as plt
import numpy as np
import math
import time

class KochSnowflake:
    def __init__(self):
        self.points = []
    
    def koch_curve(self, start, end, iterations):
        """Generate points for a Koch curve between two points"""
        if iterations == 0:
            return [start, end]
        
        # Calculate the four key points for the Koch curve
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Point 1: 1/3 of the way from start to end
        p1 = (start[0] + dx/3, start[1] + dy/3)
        
        # Point 2: 2/3 of the way from start to end
        p2 = (start[0] + 2*dx/3, start[1] + 2*dy/3)
        
        # Point 3: The peak of the equilateral triangle
        # Rotate the vector (p2-p1) by 60 degrees counterclockwise
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        # Calculate the height of the equilateral triangle
        height = math.sqrt(3) / 6 * math.sqrt(dx**2 + dy**2)
        
        # Perpendicular vector (rotated 90 degrees)
        perp_x = -dy / math.sqrt(dx**2 + dy**2) * height
        perp_y = dx / math.sqrt(dx**2 + dy**2) * height
        
        p3 = (mid_x + perp_x, mid_y + perp_y)
        
        # Recursively generate the four segments
        points = []
        points.extend(self.koch_curve(start, p1, iterations - 1)[:-1])
        points.extend(self.koch_curve(p1, p3, iterations - 1)[:-1])
        points.extend(self.koch_curve(p3, p2, iterations - 1)[:-1])
        points.extend(self.koch_curve(p2, end, iterations - 1))
        
        return points
    
    def generate_snowflake(self, iterations=4, size=1.0):
        """Generate the complete Koch snowflake"""
        # Define the three vertices of the initial equilateral triangle
        height = size * math.sqrt(3) / 2
        vertices = [
            (-size/2, -height/3),      # Bottom left
            (size/2, -height/3),       # Bottom right
            (0, 2*height/3)            # Top
        ]
        
        # Generate Koch curves for each side of the triangle
        all_points = []
        
        for i in range(3):
            start = vertices[i]
            end = vertices[(i + 1) % 3]
            curve_points = self.koch_curve(start, end, iterations)
            # Remove the last point to avoid duplication
            all_points.extend(curve_points[:-1])
        
        return all_points
    
    def save_snowflake(self, filename="koch_snowflake.png", iterations=4, size=3.0, 
                      dpi=300, figsize=(10, 10), line_width=1.0, color='blue'):
        """Generate and save the Koch snowflake as an image"""
        
        start_time = time.time()
        print(f"Generating Koch Snowflake with {iterations} iterations...")

        points = self.generate_snowflake(iterations, size)
        generation_time = time.time()
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Close the shape by adding the first point at the end
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        # Create the plot
        plt.figure(figsize=figsize, facecolor='white')
        plt.plot(x_coords, y_coords, color=color, linewidth=line_width)
        
        # Set equal aspect ratio and remove axes
        plt.axis('equal')
        plt.axis('off')
        
        # Remove margins
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save the image
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        
        end_time = time.time()
        
        print(f"Koch Snowflake saved as '{filename}'")
        print(f"Image size: {figsize[0]*dpi} x {figsize[1]*dpi} pixels")
        print(f"Iterations: {iterations}")
        print(f"Points generated: {len(points)}")
        print(f"Generation time: {generation_time - start_time:.4f} seconds")
        print(f"Total time: {end_time - start_time:.4f} seconds")
        print("-" * 50)
        
        return {
            'filename': filename,
            'points_count': len(points),
            'generation_time': generation_time - start_time,
            'total_time': end_time - start_time
        }
