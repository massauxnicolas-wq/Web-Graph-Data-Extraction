"""
Graph Structure Detector - Focused on:
1. Graph area identification
2. Axes detection
3. Legend detection and parsing
With extreme visual debugging
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import os
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for graph structure detection"""
    # Line detection
    HOUGH_THRESHOLD = 50
    MIN_LINE_LENGTH = 100
    MAX_LINE_GAP = 10
    
    # Legend detection
    LEGEND_MIN_CONTOURS = 2
    LEGEND_MAX_CONTOURS = 30
    LEGEND_MIN_AREA = 500
    LEGEND_MAX_AREA_RATIO = 0.15  # Max 15% of image area
    
    # Color clustering
    COLOR_EPS = 15
    COLOR_MIN_SAMPLES = 20
    
    # Debug
    DEBUG = True

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def create_debug_canvas(original, title="Debug View"):
    """Create a debug canvas with original image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig, ax

def draw_axes(ax, axes_data, show_labels=True):
    """Draw detected axes on the plot"""
    if axes_data['x_axis'] is not None:
        x1, y1, x2, y2 = axes_data['x_axis']
        ax.plot([x1, x2], [y1, y2], 'c-', linewidth=3, label='X-Axis Bottom')
        # Add arrows
        ax.arrow(x2, y2, 20, 0, head_width=5, head_length=10, fc='cyan', ec='cyan')
        if show_labels:
            ax.text((x1+x2)//2, y1-15, 'X-AXIS', color='cyan', fontsize=10, 
                   fontweight='bold', ha='center')
    
    if axes_data['y_axis'] is not None:
        x1, y1, x2, y2 = axes_data['y_axis']
        ax.plot([x1, x2], [y1, y2], 'm-', linewidth=3, label='Y-Axis Left')
        # Add arrows
        ax.arrow(x1, y1, 0, -20, head_width=5, head_length=10, fc='magenta', ec='magenta')
        if show_labels:
            ax.text(x1-30, (y1+y2)//2, 'Y-AXIS', color='magenta', fontsize=10,
                   fontweight='bold', va='center', rotation=90)

def draw_plot_region(ax, region, color='yellow', label='Plot Area'):
    """Draw the plot region boundary"""
    if region is not None:
        rect = Rectangle(
            (region['x_min'], region['y_min']),
            region['x_max'] - region['x_min'],
            region['y_max'] - region['y_min'],
            fill=False, edgecolor=color, linewidth=3, linestyle='--',
            label=label
        )
        ax.add_patch(rect)
        
        # Draw corner markers
        corners = [
            (region['x_min'], region['y_min']),
            (region['x_max'], region['y_min']),
            (region['x_min'], region['y_max']),
            (region['x_max'], region['y_max'])
        ]
        for cx, cy in corners:
            ax.plot(cx, cy, 'o', color=color, markersize=8, markeredgecolor='black')
        
        # Add dimensions
        width = region['x_max'] - region['x_min']
        height = region['y_max'] - region['y_min']
        ax.text(region['x_min'] + width//2, region['y_min'] - 10,
                f'{width}px', color=color, fontsize=8, ha='center')
        ax.text(region['x_min'] - 15, region['y_min'] + height//2,
                f'{height}px', color=color, fontsize=8, va='center', rotation=90)

def draw_legend_region(ax, legend_data):
    """Draw legend detection results"""
    if legend_data and legend_data['bbox'] is not None:
        x, y, w, h = legend_data['bbox']
        rect = Rectangle((x, y), w, h, fill=False, edgecolor='orange', 
                        linewidth=3, linestyle='-', label='Legend')
        ax.add_patch(rect)
        
        # Add legend entries
        if legend_data.get('text_entries'):
            y_offset = y + 15
            for entry in legend_data['text_entries'][:6]:
                ax.text(x + w + 10, y_offset, entry, fontsize=7, 
                       color='orange', fontfamily='monospace')
                y_offset += 15
        
        # Mark legend markers
        if legend_data.get('marker_boxes'):
            for marker_box in legend_data['marker_boxes'][:6]:
                mx, my, mw, mh = marker_box
                rect = Rectangle((x + mx, y + my), mw, mh, 
                               fill=False, edgecolor='lime', linewidth=1)
                ax.add_patch(rect)

# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    """Handle all image preprocessing"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        print(f"Image loaded: {self.width}x{self.height}")
        
        # Create output directory
        self.output_dir = f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save original
        cv2.imwrite(os.path.join(self.output_dir, '00_original.png'), self.original)
        
        # Preprocess
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess the image for analysis"""
        # Grayscale
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        self.denoised = cv2.bilateralFilter(self.gray, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhanced = clahe.apply(self.denoised)
        
        # Binary
        _, self.binary = cv2.threshold(
            self.enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Edges for line detection
        self.edges = cv2.Canny(self.enhanced, 50, 150)
        
        # Color information
        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        
        # Save preprocessed images
        cv2.imwrite(os.path.join(self.output_dir, '01_gray.png'), self.gray)
        cv2.imwrite(os.path.join(self.output_dir, '02_enhanced.png'), self.enhanced)
        cv2.imwrite(os.path.join(self.output_dir, '03_binary.png'), self.binary)
        cv2.imwrite(os.path.join(self.output_dir, '04_edges.png'), self.edges)

# ============================================================================
# AXIS DETECTOR
# ============================================================================

class AxisDetector:
    """Detect plot axes and graph boundaries"""
    
    def __init__(self, img_proc):
        self.img = img_proc
        self.height = img_proc.height
        self.width = img_proc.width
        
        # Detected structures
        self.axes = None
        self.plot_region = None
        self.grid_lines = []
        
    def detect(self):
        """Main detection pipeline"""
        print("\n" + "="*50)
        print("AXIS DETECTION")
        print("="*50)
        
        # Step 1: Detect all lines
        all_lines = self._detect_lines()
        print(f"Total lines detected: {len(all_lines)}")
        
        # Step 2: Classify lines (horizontal vs vertical)
        h_lines, v_lines = self._classify_lines(all_lines)
        print(f"Horizontal lines: {len(h_lines)}, Vertical lines: {len(v_lines)}")
        
        # Step 3: Find main axes (boundary lines)
        self.axes = self._find_main_axes(h_lines, v_lines)
        print(f"X-axis found: {self.axes['x_axis'] is not None}")
        print(f"Y-axis found: {self.axes['y_axis'] is not None}")
        
        # Step 4: Detect grid lines (tick marks)
        self.grid_lines = self._detect_grid_lines(h_lines, v_lines)
        print(f"Grid lines detected: {len(self.grid_lines)}")
        
        # Step 5: Determine plot region
        self.plot_region = self._determine_plot_region(h_lines, v_lines)
        print(f"Plot region: {self.plot_region}")
        
        # Step 6: Validate and refine
        self._validate_and_refine()
        
        return self.axes, self.plot_region
    
    def _detect_lines(self):
        """Detect all lines using Hough transform"""
        lines = cv2.HoughLinesP(
            self.img.edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        return [line[0] for line in lines]
    
    def _classify_lines(self, lines):
        """Classify lines as horizontal or vertical"""
        horizontal = []
        vertical = []
        
        for x1, y1, x2, y2 in lines:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle > 180:
                angle -= 180
            
            # Horizontal: near 0 or 180 degrees
            if angle < 10 or angle > 170:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                horizontal.append({
                    'line': [x1, y1, x2, y2],
                    'length': length,
                    'y_avg': (y1 + y2) / 2,
                    'x_min': min(x1, x2),
                    'x_max': max(x1, x2),
                    'angle': angle
                })
            
            # Vertical: near 90 degrees
            elif 80 < angle < 100:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                vertical.append({
                    'line': [x1, y1, x2, y2],
                    'length': length,
                    'x_avg': (x1 + x2) / 2,
                    'y_min': min(y1, y2),
                    'y_max': max(y1, y2),
                    'angle': angle
                })
        
        return horizontal, vertical
    
    def _find_main_axes(self, h_lines, v_lines):
        """Find the main axis lines that form the plot boundary"""
        axes = {
            'x_axis': None,
            'y_axis': None,
            'x_axis_bottom': None,
            'y_axis_left': None,
            'all_candidates': {
                'x': [],
                'y': []
            }
        }
        
        # Find bottom X-axis (longest horizontal line in bottom half)
        if h_lines:
            bottom_candidates = [l for l in h_lines if l['y_avg'] > self.height * 0.4]
            if bottom_candidates:
                # Sort by length, prefer longer lines
                bottom_candidates.sort(key=lambda l: l['length'], reverse=True)
                axes['x_axis'] = bottom_candidates[0]['line']
                axes['x_axis_bottom'] = bottom_candidates[0]
                
                # Store top 5 candidates for debugging
                axes['all_candidates']['x'] = bottom_candidates[:5]
        
        # Find left Y-axis (longest vertical line in left half)
        if v_lines:
            left_candidates = [l for l in v_lines if l['x_avg'] < self.width * 0.5]
            if left_candidates:
                left_candidates.sort(key=lambda l: l['length'], reverse=True)
                axes['y_axis'] = left_candidates[0]['line']
                axes['y_axis_left'] = left_candidates[0]
                
                axes['all_candidates']['y'] = left_candidates[:5]
        
        return axes
    
    def _detect_grid_lines(self, h_lines, v_lines):
        """Detect internal grid lines (not boundary axes)"""
        grid_lines = []
        
        if self.axes['x_axis'] is None or self.axes['y_axis'] is None:
            return grid_lines
        
        # Get boundary positions
        x_axis_y = (self.axes['x_axis'][1] + self.axes['x_axis'][3]) / 2
        y_axis_x = (self.axes['y_axis'][0] + self.axes['y_axis'][2]) / 2
        
        # Horizontal grid lines (between y-axis start and bottom)
        for line in h_lines:
            y_avg = line['y_avg']
            # Check if line is between top of y-axis and x-axis
            if self.axes['y_axis_left']:
                y_top = self.axes['y_axis_left']['y_min']
                if y_top < y_avg < x_axis_y - 10:  # -10 to exclude x-axis itself
                    # Check if line spans across plot area
                    if line['x_min'] > y_axis_x - 20 and line['x_max'] < self.width * 0.95:
                        grid_lines.append({
                            'type': 'grid_horizontal',
                            'line': line['line'],
                            'y_position': y_avg
                        })
        
        # Vertical grid lines (between x-axis start and right)
        for line in v_lines:
            x_avg = line['x_avg']
            if self.axes['x_axis_bottom']:
                x_start = self.axes['x_axis_bottom']['x_min']
                if y_axis_x + 10 < x_avg < x_start + self.axes['x_axis_bottom']['length']:
                    grid_lines.append({
                        'type': 'grid_vertical',
                        'line': line['line'],
                        'x_position': x_avg
                    })
        
        return grid_lines
    
    def _determine_plot_region(self, h_lines, v_lines):
        """Determine the actual plot area boundaries"""
        if self.axes['x_axis'] is None or self.axes['y_axis'] is None:
            # Fallback: use image boundaries with margins
            return {
                'x_min': int(self.width * 0.1),
                'y_min': int(self.height * 0.1),
                'x_max': int(self.width * 0.9),
                'y_max': int(self.height * 0.9)
            }
        
        # Get axis positions
        x_axis = self.axes['x_axis']
        y_axis = self.axes['y_axis']
        
        # Plot origin (where axes intersect)
        origin_x = min(y_axis[0], y_axis[2])
        origin_y = min(x_axis[1], x_axis[3])
        
        # Top boundary: Find the highest horizontal line that could be plot top
        top_candidates = [l for l in h_lines 
                         if l['y_avg'] < origin_y - 20 and l['x_min'] > origin_x - 20]
        
        if top_candidates:
            # Use the lowest top candidate (closest to plot area)
            top_y = max(l['y_avg'] for l in top_candidates)
        else:
            # Estimate from y-axis top
            if self.axes['y_axis_left']:
                top_y = self.axes['y_axis_left']['y_min']
            else:
                top_y = int(self.height * 0.1)
        
        # Right boundary: Find the rightmost vertical line
        right_candidates = [l for l in v_lines 
                           if l['x_avg'] > origin_x + 20 and l['y_min'] < origin_y - 20]
        
        if right_candidates:
            # Use the leftmost right candidate (closest to plot area)
            right_x = min(l['x_avg'] for l in right_candidates)
        else:
            # Estimate from x-axis right end
            right_x = max(x_axis[0], x_axis[2])
        
        return {
            'x_min': int(origin_x),
            'y_min': int(top_y),
            'x_max': int(right_x),
            'y_max': int(origin_y),
            'origin': (int(origin_x), int(origin_y)),
            'top_right': (int(right_x), int(top_y))
        }
    
    def _validate_and_refine(self):
        """Validate and refine the detected structures"""
        if self.plot_region is None:
            return
        
        # Ensure plot region has positive dimensions
        if self.plot_region['x_max'] <= self.plot_region['x_min']:
            self.plot_region['x_max'] = self.plot_region['x_min'] + 100
        
        if self.plot_region['y_max'] <= self.plot_region['y_min']:
            self.plot_region['y_max'] = self.plot_region['y_min'] + 100
        
        # Ensure within image bounds
        margin = 5
        self.plot_region['x_min'] = max(margin, self.plot_region['x_min'])
        self.plot_region['y_min'] = max(margin, self.plot_region['y_min'])
        self.plot_region['x_max'] = min(self.width - margin, self.plot_region['x_max'])
        self.plot_region['y_max'] = min(self.height - margin, self.plot_region['y_max'])
    
    def visualize(self, save=True):
        """Create comprehensive visualization of axis detection"""
        fig, axes_plot = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('AXIS & GRAPH REGION DETECTION ANALYSIS', 
                    fontsize=16, fontweight='bold')
        
        # 1. All detected lines with classification
        ax = axes_plot[0, 0]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('DETECTED LINES CLASSIFICATION', fontweight='bold')
        
        # Draw all lines from Hough transform
        all_lines = self._detect_lines()
        for x1, y1, x2, y2 in all_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw classified lines
        h_lines, v_lines = self._classify_lines(all_lines)
        for line_data in h_lines:
            x1, y1, x2, y2 = line_data['line']
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for line_data in v_lines:
            x1, y1, x2, y2 = line_data['line']
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        
        # Add legend manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Horizontal'),
            Line2D([0], [0], color='red', lw=2, label='Vertical'),
            Line2D([0], [0], color='green', lw=1, label='Other')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 2. Axis candidates
        ax = axes_plot[0, 1]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('AXIS CANDIDATES', fontweight='bold')
        
        if self.axes and 'all_candidates' in self.axes:
            # Draw x-axis candidates
            for i, candidate in enumerate(self.axes['all_candidates'].get('x', [])):
                x1, y1, x2, y2 = candidate['line']
                alpha = 1.0 - (i * 0.2)
                ax.plot([x1, x2], [y1, y2], 'c-', linewidth=3, alpha=alpha,
                       label=f'X-Candidate {i+1}')
            
            # Draw y-axis candidates
            for i, candidate in enumerate(self.axes['all_candidates'].get('y', [])):
                x1, y1, x2, y2 = candidate['line']
                alpha = 1.0 - (i * 0.2)
                ax.plot([x1, x2], [y1, y2], 'm-', linewidth=3, alpha=alpha,
                       label=f'Y-Candidate {i+1}')
        
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # 3. Selected main axes
        ax = axes_plot[0, 2]
        img = self.img.original.copy()
        
        if self.axes:
            # Draw selected axes
            if self.axes['x_axis'] is not None:
                x1, y1, x2, y2 = self.axes['x_axis']
                ax.plot([x1, x2], [y1, y2], 'cyan', linewidth=4, label='X-Axis')
                ax.plot(x1, y1, 'co', markersize=10)
                ax.plot(x2, y2, 'co', markersize=10)
            
            if self.axes['y_axis'] is not None:
                x1, y1, x2, y2 = self.axes['y_axis']
                ax.plot([x1, x2], [y1, y2], 'magenta', linewidth=4, label='Y-Axis')
                ax.plot(x1, y1, 'mo', markersize=10)
                ax.plot(x2, y2, 'mo', markersize=10)
        
        # Draw grid lines
        for grid_line in self.grid_lines:
            x1, y1, x2, y2 = grid_line['line']
            if grid_line['type'] == 'grid_horizontal':
                ax.plot([x1, x2], [y1, y2], 'blue', linewidth=1, alpha=0.5)
            else:
                ax.plot([x1, x2], [y1, y2], 'blue', linewidth=1, alpha=0.5)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('MAIN AXES & GRID', fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # 4. Plot region determination
        ax = axes_plot[1, 0]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('PLOT REGION DETERMINATION', fontweight='bold')
        
        if self.plot_region:
            pr = self.plot_region
            # Draw the plot region
            rect = Rectangle(
                (pr['x_min'], pr['y_min']),
                pr['x_max'] - pr['x_min'],
                pr['y_max'] - pr['y_min'],
                fill=True, facecolor='yellow', alpha=0.2, edgecolor='yellow',
                linewidth=3, linestyle='--'
            )
            ax.add_patch(rect)
            
            # Draw corners
            for label, (cx, cy) in {
                'Origin': (pr['x_min'], pr['y_max']),
                'Top-Left': (pr['x_min'], pr['y_min']),
                'Top-Right': (pr['x_max'], pr['y_min']),
                'Bottom-Right': (pr['x_max'], pr['y_max'])
            }.items():
                ax.plot(cx, cy, 'o', color='orange', markersize=10, 
                       markeredgecolor='black')
                ax.text(cx + 10, cy - 10, label, color='orange', fontsize=8,
                       fontweight='bold')
        
        ax.axis('off')
        
        # 5. Region validation
        ax = axes_plot[1, 1]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('REGION ANALYSIS', fontweight='bold')
        
        if self.plot_region:
            pr = self.plot_region
            
            # Analyze content in different regions
            # Left of y-axis (axis labels area)
            if pr['x_min'] > 10:
                label_region = self.img.binary[pr['y_min']:pr['y_max'], 0:pr['x_min']]
                label_density = np.sum(label_region > 0) / label_region.size if label_region.size > 0 else 0
                
                rect = Rectangle((0, pr['y_min']), pr['x_min'], 
                               pr['y_max'] - pr['y_min'],
                               fill=True, facecolor='red', alpha=0.2)
                ax.add_patch(rect)
                ax.text(pr['x_min']//2, (pr['y_min']+pr['y_max'])//2,
                       f'Y-Labels\n{label_density:.1%}',
                       color='red', fontsize=8, ha='center')
            
            # Below x-axis (x-axis labels area)
            if pr['y_max'] < self.height - 10:
                label_region = self.img.binary[pr['y_max']:self.height, pr['x_min']:pr['x_max']]
                label_density = np.sum(label_region > 0) / label_region.size if label_region.size > 0 else 0
                
                rect = Rectangle((pr['x_min'], pr['y_max']),
                               pr['x_max'] - pr['x_min'],
                               self.height - pr['y_max'],
                               fill=True, facecolor='blue', alpha=0.2)
                ax.add_patch(rect)
                ax.text((pr['x_min']+pr['x_max'])//2, 
                       (pr['y_max']+self.height)//2,
                       f'X-Labels\n{label_density:.1%}',
                       color='blue', fontsize=8, ha='center')
            
            # Title area (above plot)
            if pr['y_min'] > 10:
                title_region = self.img.binary[0:pr['y_min'], pr['x_min']:pr['x_max']]
                title_density = np.sum(title_region > 0) / title_region.size if title_region.size > 0 else 0
                
                rect = Rectangle((pr['x_min'], 0),
                               pr['x_max'] - pr['x_min'],
                               pr['y_min'],
                               fill=True, facecolor='green', alpha=0.2)
                ax.add_patch(rect)
                ax.text((pr['x_min']+pr['x_max'])//2, pr['y_min']//2,
                       f'Title?\n{title_density:.1%}',
                       color='green', fontsize=8, ha='center')
            
            # Right of plot (legend area?)
            if pr['x_max'] < self.width - 10:
                legend_region = self.img.binary[pr['y_min']:pr['y_max'], 
                                               pr['x_max']:self.width]
                legend_density = np.sum(legend_region > 0) / legend_region.size if legend_region.size > 0 else 0
                
                rect = Rectangle((pr['x_max'], pr['y_min']),
                               self.width - pr['x_max'],
                               pr['y_max'] - pr['y_min'],
                               fill=True, facecolor='purple', alpha=0.2)
                ax.add_patch(rect)
                ax.text((pr['x_max']+self.width)//2, 
                       (pr['y_min']+pr['y_max'])//2,
                       f'Legend?\n{legend_density:.1%}',
                       color='purple', fontsize=8, ha='center')
        
        ax.axis('off')
        
        # 6. Statistics
        ax = axes_plot[1, 2]
        ax.axis('off')
        
        stats = f"""
        AXIS DETECTION STATISTICS
        
        Image Size: {self.width}x{self.height}px
        
        MAIN AXES:
        X-Axis Bottom: {self.axes['x_axis'] is not None}
        Y-Axis Left: {self.axes['y_axis'] is not None}
        
        """
        
        if self.axes['x_axis'] is not None:
            x1, y1, x2, y2 = self.axes['x_axis']
            stats += f"""
        X-Axis: ({x1},{y1}) → ({x2},{y2})
        Length: {np.sqrt((x2-x1)**2 + (y2-y1)**2):.0f}px
        """
        
        if self.axes['y_axis'] is not None:
            x1, y1, x2, y2 = self.axes['y_axis']
            stats += f"""
        Y-Axis: ({x1},{y1}) → ({x2},{y2})
        Length: {np.sqrt((x2-x1)**2 + (y2-y1)**2):.0f}px
        """
        
        if self.plot_region:
            pr = self.plot_region
            stats += f"""
        PLOT REGION:
        Width: {pr['x_max'] - pr['x_min']}px
        Height: {pr['y_max'] - pr['y_min']}px
        Area: {(pr['x_max'] - pr['x_min']) * (pr['y_max'] - pr['y_min'])}px²
        
        Grid Lines: {len(self.grid_lines)}
        """
        
        ax.text(0.1, 0.9, stats, fontfamily='monospace', fontsize=10,
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.img.output_dir, '05_axis_analysis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()
        return fig

# ============================================================================
# LEGEND DETECTOR
# ============================================================================

class LegendDetector:
    """Advanced legend detection and parsing"""
    
    def __init__(self, img_proc, plot_region):
        self.img = img_proc
        self.plot_region = plot_region
        self.legend_data = None
        self.candidates = []
        
    def detect(self):
        """Detect and parse legend"""
        print("\n" + "="*50)
        print("LEGEND DETECTION")
        print("="*50)
        
        # Step 1: Find candidates in multiple locations
        self.candidates = self._find_all_candidates()
        print(f"Candidates found: {len(self.candidates)}")
        
        if not self.candidates:
            print("No legend candidates found")
            return None
        
        # Step 2: Score and rank candidates
        self.candidates = self._score_candidates(self.candidates)
        
        # Step 3: Select best candidate
        best = self.candidates[0]
        print(f"Best candidate score: {best['score']:.2f}")
        print(f"Location: {best['location']}")
        
        # Step 4: Parse legend content
        self.legend_data = self._parse_legend_content(best)
        
        if self.legend_data:
            print(f"Legend entries found: {len(self.legend_data.get('text_entries', []))}")
        
        return self.legend_data
    
    def _find_all_candidates(self):
        """Search for legend in multiple locations"""
        candidates = []
        pr = self.plot_region
        
        if pr is None:
            return candidates
        
        # Define search zones
        search_zones = [
            {
                'name': 'inside_top_right',
                'x1': pr['x_max'] - 200,
                'y1': pr['y_min'],
                'x2': pr['x_max'],
                'y2': pr['y_min'] + 150,
                'expected': 'Legend inside plot (top-right)'
            },
            {
                'name': 'inside_top_left',
                'x1': pr['x_min'],
                'y1': pr['y_min'],
                'x2': pr['x_min'] + 200,
                'y2': pr['y_min'] + 150,
                'expected': 'Legend inside plot (top-left)'
            },
            {
                'name': 'outside_right',
                'x1': pr['x_max'] + 5,
                'y1': pr['y_min'] + 20,
                'x2': min(self.img.width, pr['x_max'] + 200),
                'y2': min(self.img.height, pr['y_max'] - 20),
                'expected': 'Legend outside plot (right)'
            },
            {
                'name': 'outside_bottom',
                'x1': pr['x_min'],
                'y1': pr['y_max'] + 5,
                'x2': pr['x_max'],
                'y2': min(self.img.height, pr['y_max'] + 80),
                'expected': 'Legend below plot'
            }
        ]
        
        for zone in search_zones:
            # Validate zone coordinates
            x1 = max(0, zone['x1'])
            y1 = max(0, zone['y1'])
            x2 = min(self.img.width, zone['x2'])
            y2 = min(self.img.height, zone['y2'])
            
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue
            
            # Extract region
            roi_color = self.img.original[y1:y2, x1:x2]
            roi_binary = self.img.binary[y1:y2, x1:x2]
            roi_edges = self.img.edges[y1:y2, x1:x2]
            
            # Analyze region
            features = self._extract_region_features(
                roi_color, roi_binary, roi_edges, zone['name']
            )
            
            if features is not None:
                candidates.append({
                    'zone': zone,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'features': features,
                    'score': 0
                })
        
        return candidates
    
    def _extract_region_features(self, color_roi, binary_roi, edges_roi, zone_name):
        """Extract features that indicate a legend"""
        features = {}
        
        # 1. Contour analysis
        contours, hierarchy = cv2.findContours(
            binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
        
        features['num_contours'] = len(significant_contours)
        features['total_contour_area'] = sum(cv2.contourArea(c) for c in significant_contours)
        
        # 2. Color analysis
        colors = self._extract_colors(color_roi)
        features['num_colors'] = len(colors)
        features['has_distinct_colors'] = len(colors) >= 2
        
        # 3. Text density
        text_density = np.sum(binary_roi > 0) / binary_roi.size if binary_roi.size > 0 else 0
        features['text_density'] = text_density
        
        # 4. Shape analysis - look for small colored markers
        hsv_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        
        # Detect small colored rectangles/circles (legend markers)
        marker_like = self._detect_markers(color_roi, contours)
        features['marker_count'] = len(marker_like)
        features['marker_boxes'] = marker_like
        
        # 5. Check if region has text-like patterns
        # (alternating text and markers)
        features['has_text_pattern'] = (
            features['text_density'] > 0.05 and 
            features['num_contours'] > 2
        )
        
        # Basic filter: must have some content
        if features['num_contours'] < 2:
            return None
        
        return features
    
    def _extract_colors(self, roi):
        """Extract distinct colors from a region"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create saturation mask
        sat_mask = hsv[:, :, 1] > 30
        val_mask = hsv[:, :, 2] > 30
        val_mask2 = hsv[:, :, 2] < 225
        
        mask = sat_mask & val_mask & val_mask2
        
        if np.sum(mask) < 50:
            return []
        
        pixels = hsv[mask]
        
        # Cluster colors
        clustering = DBSCAN(eps=15, min_samples=20).fit(pixels)
        
        colors = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster = pixels[clustering.labels_ == label]
                if len(cluster) > 20:
                    avg_color = np.median(cluster, axis=0)
                    colors.append(avg_color)
        
        return colors
    
    def _detect_markers(self, color_roi, contours):
        """Detect small colored markers (could be legend markers)"""
        markers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Legend markers are usually small squares or circles
            if 20 < area < 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if shape is roughly square or circle
                if 0.5 < aspect_ratio < 2.0:
                    markers.append((x, y, w, h))
        
        return markers
    
    def _score_candidates(self, candidates):
        """Score candidates based on legend-like features"""
        for candidate in candidates:
            features = candidate['features']
            score = 0
            
            # More contours suggests more legend entries (but not too many)
            if 2 <= features['num_contours'] <= 15:
                score += 3
            elif 16 <= features['num_contours'] <= 30:
                score += 1
            
            # Having distinct colors is a good sign
            if features['has_distinct_colors']:
                score += 4
            
            # Number of colors (legends usually have 2-8 entries)
            if 2 <= features['num_colors'] <= 8:
                score += 3
            
            # Text density should be moderate
            if 0.03 <= features['text_density'] <= 0.3:
                score += 2
            
            # Marker-like shapes
            if features['marker_count'] >= 2:
                score += 4
            elif features['marker_count'] >= 1:
                score += 2
            
            # Text pattern
            if features['has_text_pattern']:
                score += 3
            
            # Location preference (outside right is most common)
            if candidate['zone']['name'] == 'outside_right':
                score += 1
            
            candidate['score'] = score
        
        # Sort by score
        candidates.sort(key=lambda c: c['score'], reverse=True)
        
        return candidates
    
    def _parse_legend_content(self, candidate):
        """Parse the content of the detected legend"""
        x, y, w, h = candidate['bbox']
        roi = self.img.original[y:y+h, x:x+w]
        
        # Try to segment legend into entries
        entries = self._segment_legend_entries(roi, candidate['features'])
        
        # Extract text from each entry
        text_entries = []
        for entry_roi in entries:
            text = self._ocr_text(entry_roi)
            if text:
                text_entries.append(text)
        
        return {
            'bbox': candidate['bbox'],
            'location': candidate['zone']['name'],
            'score': candidate['score'],
            'text_entries': text_entries,
            'marker_boxes': candidate['features'].get('marker_boxes', []),
            'num_colors': candidate['features']['num_colors'],
            'confidence': min(1.0, candidate['score'] / 15)
        }
    
    def _segment_legend_entries(self, legend_roi, features):
        """Segment legend into individual entries"""
        gray = cv2.cvtColor(legend_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find text contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Group contours by y-coordinate (entries are usually horizontal)
        entries = []
        if contours:
            # Sort by y position
            bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
            bounding_boxes.sort(key=lambda b: b[1])  # Sort by y
            
            # Group into rows
            current_row = []
            current_y = None
            
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                
                if current_y is None or abs(y - current_y) < 20:
                    current_row.append(bbox)
                else:
                    if current_row:
                        # Merge current row
                        entries.append(self._merge_row_boxes(current_row))
                    current_row = [bbox]
                
                current_y = y
            
            if current_row:
                entries.append(self._merge_row_boxes(current_row))
        
        # Extract ROI for each entry
        entry_rois = []
        for (ex, ey, ew, eh) in entries:
            if ey < legend_roi.shape[0] and ex < legend_roi.shape[1]:
                entry_roi = legend_roi[max(0, ey):min(legend_roi.shape[0], ey+eh+5),
                                      max(0, ex):min(legend_roi.shape[1], ex+ew+5)]
                if entry_roi.size > 0:
                    entry_rois.append(entry_roi)
        
        return entry_rois
    
    def _merge_row_boxes(self, boxes):
        """Merge bounding boxes in the same row"""
        if not boxes:
            return (0, 0, 0, 0)
        
        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[0] + b[2] for b in boxes)
        y_max = max(b[1] + b[3] for b in boxes)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _ocr_text(self, roi):
        """Extract text from ROI"""
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 7').strip()
            return text if text else None
        except:
            return None
    
    def visualize(self, save=True):
        """Create visualization of legend detection"""
        fig, axes_plot = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('LEGEND DETECTION ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. All search zones
        ax = axes_plot[0, 0]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('LEGEND SEARCH ZONES', fontweight='bold')
        
        zone_colors = {
            'inside_top_right': 'green',
            'inside_top_left': 'blue',
            'outside_right': 'orange',
            'outside_bottom': 'red'
        }
        
        for zone_name, color in zone_colors.items():
            for cand in self.candidates:
                if cand['zone']['name'] == zone_name:
                    x, y, w, h = cand['bbox']
                    rect = Rectangle((x, y), w, h, fill=True, facecolor=color, 
                                   alpha=0.3, edgecolor=color, linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + w//2, y + h//2, zone_name, fontsize=7, 
                           ha='center', va='center', color='white', fontweight='bold')
                    break
            else:
                # Draw empty zone if no candidate found
                # (zones defined in _find_all_candidates)
                pass
        
        ax.axis('off')
        
        # 2. Candidates scoring
        ax = axes_plot[0, 1]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('CANDIDATES RANKED BY SCORE', fontweight='bold')
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 1, max(1, len(self.candidates))))
        
        for i, cand in enumerate(self.candidates[:5]):
            x, y, w, h = cand['bbox']
            color = colors[i] if i < len(colors) else 'gray'
            rect = Rectangle((x, y), w, h, fill=True, facecolor=color, 
                           alpha=0.3, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + 5, y + 15, f"#{i+1} Score:{cand['score']:.1f}", 
                   fontsize=7, color='white', fontweight='bold')
        
        ax.axis('off')
        
        # 3. Best candidate detail
        ax = axes_plot[0, 2]
        img = self.img.original.copy()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('BEST CANDIDATE DETAIL', fontweight='bold')
        
        if self.candidates:
            best = self.candidates[0]
            x, y, w, h = best['bbox']
            
            # Highlight the region
            rect = Rectangle((x, y), w, h, fill=True, facecolor='yellow', 
                           alpha=0.2, edgecolor='yellow', linewidth=3)
            ax.add_patch(rect)
            
            # Draw markers
            for (mx, my, mw, mh) in best['features'].get('marker_boxes', [])[:10]:
                rect = Rectangle((x + mx, y + my), mw, mh, 
                               fill=False, edgecolor='lime', linewidth=1)
                ax.add_patch(rect)
        
        ax.axis('off')
        
        # 4. Feature analysis
        ax = axes_plot[1, 0]
        ax.axis('off')
        
        if self.candidates:
            best = self.candidates[0]
            features = best['features']
            
            feature_text = f"""
            BEST CANDIDATE FEATURES
            
            Location: {best['zone']['name']}
            Score: {best['score']:.2f}
            
            Contours: {features['num_contours']}
            Colors detected: {features['num_colors']}
            Distinct colors: {features['has_distinct_colors']}
            Markers found: {features['marker_count']}
            Text density: {features['text_density']:.3f}
            Text pattern: {features['has_text_pattern']}
            
            Size: {best['bbox'][2]}x{best['bbox'][3]}px
            """
            
            ax.text(0.1, 0.9, feature_text, fontfamily='monospace', fontsize=9,
                   verticalalignment='top', transform=ax.transAxes)
        
        # 5. Legend parsing
        ax = axes_plot[1, 1]
        ax.axis('off')
        
        if self.legend_data:
            legend_text = "PARSED LEGEND CONTENT:\n\n"
            
            for i, entry in enumerate(self.legend_data.get('text_entries', [])[:10]):
                legend_text += f"{i+1}. {entry}\n"
            
            legend_text += f"\nConfidence: {self.legend_data['confidence']:.2%}"
            
            ax.text(0.1, 0.9, legend_text, fontfamily='monospace', fontsize=9,
                   verticalalignment='top', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'NO LEGEND PARSED', ha='center', va='center',
                   fontsize=14, fontweight='bold')
        
        # 6. All candidates comparison
        ax = axes_plot[1, 2]
        
        if self.candidates:
            names = [c['zone']['name'][:15] for c in self.candidates[:6]]
            scores = [c['score'] for c in self.candidates[:6]]
            colors_bar = plt.cm.RdYlGn(np.array(scores) / max(scores) if max(scores) > 0 else 0)
            
            bars = ax.bar(range(len(names)), scores, color=colors_bar)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Score')
            ax.set_title('CANDIDATE SCORES', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{score:.1f}', ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No candidates found', ha='center', va='center')
            ax.set_title('CANDIDATE SCORES', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.img.output_dir, '06_legend_analysis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()
        return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run graph structure detection"""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please provide an image path")
        print("Usage: python graph_structure_detector.py <image_path>")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print("\n" + "="*70)
    print("GRAPH STRUCTURE DETECTOR")
    print("Focused on: Graph Area, Axes, and Legend Detection")
    print("="*70)
    
    # Process image
    img_proc = ImageProcessor(image_path)
    
    # Detect axes and plot region
    axis_detector = AxisDetector(img_proc)
    axes, plot_region = axis_detector.detect()
    
    # Visualize axis detection
    axis_detector.visualize()
    
    # Detect legend
    legend_detector = LegendDetector(img_proc, plot_region)
    legend_data = legend_detector.detect()
    
    # Visualize legend detection
    if legend_data or legend_detector.candidates:
        legend_detector.visualize()
    
    # Create final summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    img = img_proc.original.copy()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw axes
    draw_axes(ax, axes)
    
    # Draw plot region
    draw_plot_region(ax, plot_region)
    
    # Draw legend
    draw_legend_region(ax, legend_data)
    
    ax.set_title('GRAPH STRUCTURE SUMMARY', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.axis('off')
    
    # Save summary
    summary_path = os.path.join(img_proc.output_dir, '07_structure_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {summary_path}")
    plt.show()
    
    # Save results to JSON
    results = {
        'image': image_path,
        'image_size': {'width': img_proc.width, 'height': img_proc.height},
        'axes': {
            'x_axis': axes['x_axis'] if axes['x_axis'] else None,
            'y_axis': axes['y_axis'] if axes['y_axis'] else None
        },
        'plot_region': plot_region,
        'legend': legend_data
    }
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results = convert(results)
    
    json_path = os.path.join(img_proc.output_dir, 'structure_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    print(f"All outputs in: {img_proc.output_dir}")
    print("\nDone!")

if __name__ == "__main__":
    main()