"""
AutoPlot Digitizer MVP - Complete Automated Data Extraction
With Extreme Visual Debugging
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import pytesseract
from scipy import ndimage
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the extractor"""
    # Image processing
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    # Axis detection
    AXIS_MIN_LENGTH_RATIO = 0.3  # Minimum axis length relative to image
    
    # Tick detection
    TICK_MIN_LENGTH = 5
    TICK_MAX_LENGTH = 40
    TICK_ANGLE_TOLERANCE = 5  # degrees
    
    # OCR
    TESSERACT_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789.-+eE×'
    
    # Curve extraction
    CURVE_COLOR_THRESHOLD = 30
    MIN_CURVE_POINTS = 20
    
    # Debug
    DEBUG_LEVEL = 'extreme'  # 'basic', 'detailed', 'extreme'
    SAVE_INTERMEDIATE = True

# ============================================================================
# IMAGE PREPROCESSOR
# ============================================================================

class ImagePreprocessor:
    """Handle all image preprocessing operations"""
    
    def __init__(self, config):
        self.config = config
        self.original = None
        self.gray = None
        self.denoised = None
        self.enhanced = None
        self.binary = None
        self.edges = None
        self.height = 0
        self.width = 0
    
    def preprocess(self, image_path):
        """Full preprocessing pipeline"""
        # Load image
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Store dimensions
        self.height, self.width = self.original.shape[:2]
        
        # Create debug directory
        self.debug_dir = 'debug_outputs'
        os.makedirs(self.debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.debug_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save original
        cv2.imwrite(os.path.join(self.run_dir, '00_original.png'), self.original)
        
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        self.denoised = cv2.bilateralFilter(self.gray, 9, 75, 75)
        cv2.imwrite(os.path.join(self.run_dir, '01_denoised.png'), self.denoised)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
        self.enhanced = clahe.apply(self.denoised)
        cv2.imwrite(os.path.join(self.run_dir, '02_enhanced.png'), self.enhanced)
        
        # Binarize
        _, self.binary = cv2.threshold(
            self.enhanced, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cv2.imwrite(os.path.join(self.run_dir, '03_binary.png'), self.binary)
        
        # Edge detection
        self.edges = cv2.Canny(self.enhanced, 50, 150)
        cv2.imwrite(os.path.join(self.run_dir, '04_edges.png'), self.edges)
        
        return self

# ============================================================================
# AXIS DETECTOR
# ============================================================================

class AxisDetector:
    """Detect plot axes using Hough transform"""
    
    def __init__(self, preprocessor):
        self.prep = preprocessor
        self.config = preprocessor.config
        self.image = preprocessor.original.copy()
        self.height = preprocessor.height
        self.width = preprocessor.width
        
    def detect(self):
        """Main axis detection pipeline"""
        # Detect lines
        lines = self._detect_lines()
        
        # Cluster lines
        horizontal_lines, vertical_lines = self._cluster_lines(lines)
        
        # Find main axes
        self.axes = self._find_main_axes(horizontal_lines, vertical_lines)
        
        # Refine plot region to exclude axis labels and legends
        self._refine_plot_region()
        
        # Create debug visualization
        self._create_debug_visualization(lines, horizontal_lines, vertical_lines)
        
        return self.axes
    
    def _detect_lines(self):
        """Detect lines using Hough transform"""
        lines = cv2.HoughLinesP(
            self.prep.edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        return [line[0] for line in lines]
    
    def _cluster_lines(self, lines):
        """Cluster lines into horizontal and vertical groups"""
        if not lines:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Normalize angle to [0, 180]
            if angle > 180:
                angle = angle - 180
            
            # Horizontal lines (angle near 0 or 180)
            if angle < 10 or angle > 170:
                horizontal_lines.append([x1, y1, x2, y2])
            
            # Vertical lines (angle near 90)
            elif 80 < angle < 100:
                vertical_lines.append([x1, y1, x2, y2])
        
        return horizontal_lines, vertical_lines
    
    def _find_main_axes(self, horizontal_lines, vertical_lines):
        """Find the main axis lines that form the plot boundary"""
        axes = {
            'x_axis': None,
            'y_axis': None,
            'plot_region': None
        }
        
        # Find longest horizontal line near bottom
        if horizontal_lines:
            # Sort by length
            horizontal_lines.sort(
                key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2),
                reverse=True
            )
            
            # Take lines in bottom third
            bottom_lines = [
                l for l in horizontal_lines 
                if l[1] > self.height * 0.5 and l[3] > self.height * 0.5
            ]
            
            if bottom_lines:
                axes['x_axis'] = bottom_lines[0]
        
        # Find longest vertical line near left
        if vertical_lines:
            vertical_lines.sort(
                key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2),
                reverse=True
            )
            
            left_lines = [
                l for l in vertical_lines 
                if l[0] < self.width * 0.5 and l[2] < self.width * 0.5
            ]
            
            if left_lines:
                axes['y_axis'] = left_lines[0]
        
        # Find plot region
        if axes['x_axis'] is not None and axes['y_axis'] is not None:
            x_axis = axes['x_axis']
            y_axis = axes['y_axis']
            
            # Find intersection (plot origin)
            x_min = min(y_axis[0], y_axis[2])
            y_max = min(x_axis[1], x_axis[3])
            
            # Find top boundary (search for horizontal lines above the plot)
            top_lines = [
                l for l in horizontal_lines 
                if l[1] < y_max - 50 and l[1] > 10
            ]
            if top_lines:
                # Use the lowest top line (closest to plot area)
                y_min = max([min(l[1], l[3]) for l in top_lines])
            else:
                # Fallback: use 10% from top
                y_min = int(self.height * 0.1)
            
            # Find right boundary (search for vertical lines right of the plot)
            right_lines = [
                l for l in vertical_lines 
                if l[0] > x_min + 50 and l[0] < self.width - 50
            ]
            if right_lines:
                # Use the leftmost right line (closest to plot area)
                x_max = min([min(l[0], l[2]) for l in right_lines])
            else:
                # Fallback: use 90% of width
                x_max = int(self.width * 0.9)
            
            axes['plot_region'] = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            }
        else:
            # Fallback: use entire image with margins
            axes['plot_region'] = {
                'x_min': int(self.width * 0.1),
                'y_min': int(self.height * 0.1),
                'x_max': int(self.width * 0.9),
                'y_max': int(self.height * 0.9)
            }
        
        return axes
    
    def _refine_plot_region(self):
        """Refine plot region to avoid including axes labels and legends"""
        if self.axes['plot_region'] is None:
            return
        
        pr = self.axes['plot_region']
        
        # Expand region slightly to ensure we capture the full plot area
        margin = 10
        
        # Check for text/legend on the right side
        # If right side has significant white space or text, shrink the region
        right_roi = self.prep.binary[
            pr['y_min']:pr['y_max'],
            pr['x_max']:min(self.width, pr['x_max'] + 100)
        ]
        
        if right_roi.size > 0:
            # Check if there's text/features on the right
            white_pixels = np.sum(right_roi > 0)
            if white_pixels > 100:  # If there's content on the right
                # This might be legend or labels, keep current boundary
                pass
            else:
                # Empty space, can extend slightly
                pr['x_max'] = min(self.width - margin, pr['x_max'] + 20)
        
        # Ensure plot region is within image bounds
        pr['x_min'] = max(margin, pr['x_min'])
        pr['y_min'] = max(margin, pr['y_min'])
        pr['x_max'] = min(self.width - margin, pr['x_max'])
        pr['y_max'] = min(self.height - margin, pr['y_max'])
    
    def _create_debug_visualization(self, all_lines, horizontal_lines, vertical_lines):
        """Create extreme visual debugging for axis detection"""
        # Create multiple debug views
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('AXIS DETECTION - EXTREME DEBUG', fontsize=16, fontweight='bold')
        
        # 1. All detected lines
        ax = axes[0, 0]
        img = self.image.copy()
        for x1, y1, x2, y2 in all_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'ALL DETECTED LINES ({len(all_lines)})', fontweight='bold')
        ax.axis('off')
        
        # 2. Horizontal lines
        ax = axes[0, 1]
        img = self.image.copy()
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'HORIZONTAL LINES ({len(horizontal_lines)})', fontweight='bold')
        ax.axis('off')
        
        # 3. Vertical lines
        ax = axes[0, 2]
        img = self.image.copy()
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'VERTICAL LINES ({len(vertical_lines)})', fontweight='bold')
        ax.axis('off')
        
        # 4. Main axes detection
        ax = axes[1, 0]
        img = self.image.copy()
        if self.axes['x_axis'] is not None:
            x1, y1, x2, y2 = self.axes['x_axis']
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(img, 'X-AXIS', ((x1+x2)//2, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.axes['y_axis'] is not None:
            x1, y1, x2, y2 = self.axes['y_axis']
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, 'Y-AXIS', (x1-80, (y1+y2)//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('MAIN AXES', fontweight='bold')
        ax.axis('off')
        
        # 5. Plot region
        ax = axes[1, 1]
        img = self.image.copy()
        if self.axes['plot_region']:
            pr = self.axes['plot_region']
            cv2.rectangle(img, 
                         (pr['x_min'], pr['y_min']),
                         (pr['x_max'], pr['y_max']),
                         (0, 255, 0), 2)
            # Draw corners
            for x, y in [(pr['x_min'], pr['y_min']), 
                        (pr['x_max'], pr['y_min']),
                        (pr['x_min'], pr['y_max']),
                        (pr['x_max'], pr['y_max'])]:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # Label the region
            cv2.putText(img, 'PLOT AREA', (pr['x_min']+5, pr['y_min']+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('PLOT REGION', fontweight='bold')
        ax.axis('off')
        
        # 6. Axis statistics
        ax = axes[1, 2]
        ax.axis('off')
        info_text = f"""
        AXIS DETECTION STATISTICS:
        
        Total lines detected: {len(all_lines)}
        Horizontal lines: {len(horizontal_lines)}
        Vertical lines: {len(vertical_lines)}
        
        X-AXIS FOUND: {self.axes['x_axis'] is not None}
        Y-AXIS FOUND: {self.axes['y_axis'] is not None}
        
        PLOT REGION DETECTED: {self.axes['plot_region'] is not None}
        """
        if self.axes['plot_region']:
            pr = self.axes['plot_region']
            info_text += f"""
        Plot Region:
        X: {pr['x_min']} to {pr['x_max']}
        Y: {pr['y_min']} to {pr['y_max']}
        Width: {pr['x_max'] - pr['x_min']}px
        Height: {pr['y_max'] - pr['y_min']}px
        
        Image Size: {self.width}x{self.height}
        """
        
        ax.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=10,
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.prep.run_dir, '05_axis_detection_debug.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# TICK DETECTOR
# ============================================================================

class TickDetector:
    """Detect and read tick marks and labels"""
    
    def __init__(self, preprocessor, axes):
        self.prep = preprocessor
        self.axes = axes
        self.config = preprocessor.config
        self.image = preprocessor.original.copy()
        self.height = preprocessor.height
        self.width = preprocessor.width
        
    def detect_ticks(self):
        """Detect tick marks and their labels"""
        self.ticks = {
            'x_ticks': [],
            'y_ticks': []
        }
        
        pr = self.axes['plot_region']
        
        # Detect X-axis ticks
        if self.axes['x_axis'] is not None:
            self._detect_axis_ticks('x', pr)
        
        # Detect Y-axis ticks
        if self.axes['y_axis'] is not None:
            self._detect_axis_ticks('y', pr)
        
        # Create debug visualization
        self._create_debug_visualization()
        
        return self.ticks
    
    def _detect_axis_ticks(self, axis_type, plot_region):
        """Detect ticks for a specific axis"""
        if axis_type == 'x':
            axis_line = self.axes['x_axis']
            tick_length_range = (5, 30)
            search_region_expand = 30
        else:
            axis_line = self.axes['y_axis']
            tick_length_range = (5, 30)
            search_region_expand = 50
        
        # Get axis position and orientation
        x1, y1, x2, y2 = axis_line
        
        # Search for tick marks perpendicular to axis
        tick_marks = self._find_tick_marks(
            axis_line, tick_length_range, search_region_expand, axis_type
        )
        
        # Match ticks with text labels
        for tick_pos in tick_marks:
            tick = {
                'position': tick_pos,
                'label': None,
                'value': None,
                'confidence': 0
            }
            
            # Try to read label
            label_data = self._read_tick_label(tick_pos, axis_type)
            if label_data:
                tick.update(label_data)
            
            self.ticks[f'{axis_type}_ticks'].append(tick)
    
    def _find_tick_marks(self, axis_line, length_range, search_expand, axis_type):
        """Find tick marks along an axis"""
        x1, y1, x2, y2 = axis_line
        
        # Create search region
        if axis_type == 'x':
            # Search below the axis for vertical tick marks
            roi_y1 = y1 - 5
            roi_y2 = y1 + search_expand
            roi_x1 = x1 - 10
            roi_x2 = x2 + 10
        else:
            # Search left of the axis for horizontal tick marks
            roi_x1 = x1 - search_expand
            roi_x2 = x1 + 5
            roi_y1 = y1 - 10
            roi_y2 = y2 + 10
        
        # Ensure ROI is within image bounds
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(self.width, roi_x2)
        roi_y2 = min(self.height, roi_y2)
        
        # Check if ROI is valid
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return []
        
        # Extract ROI
        roi = self.prep.edges[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Detect lines in ROI (these are potential tick marks)
        tick_lines = cv2.HoughLinesP(
            roi, 1, np.pi/180, 
            threshold=15,
            minLineLength=length_range[0],
            maxLineGap=2
        )
        
        tick_positions = []
        if tick_lines is not None:
            for line in tick_lines:
                tx1, ty1, tx2, ty2 = line[0]
                
                # Convert to image coordinates
                tx1 += roi_x1
                ty1 += roi_y1
                tx2 += roi_x1
                ty2 += roi_y1
                
                # Check if line is perpendicular to axis
                angle = np.abs(np.arctan2(ty2 - ty1, tx2 - tx1) * 180 / np.pi)
                
                if axis_type == 'x':
                    # Looking for vertical ticks (angle near 90)
                    if 80 < angle < 100:
                        tick_positions.append((tx1 + tx2) // 2)
                else:
                    # Looking for horizontal ticks (angle near 0 or 180)
                    if angle < 10 or angle > 170:
                        tick_positions.append((ty1 + ty2) // 2)
        
        # Sort and deduplicate
        tick_positions = sorted(set(tick_positions))
        
        # Cluster nearby ticks
        if tick_positions:
            clustered = []
            current_cluster = [tick_positions[0]]
            
            for i in range(1, len(tick_positions)):
                if tick_positions[i] - tick_positions[i-1] < 10:
                    current_cluster.append(tick_positions[i])
                else:
                    clustered.append(int(np.mean(current_cluster)))
                    current_cluster = [tick_positions[i]]
            
            clustered.append(int(np.mean(current_cluster)))
            return clustered
        
        return []
    
    def _read_tick_label(self, tick_position, axis_type):
        """Read the text label near a tick mark"""
        if axis_type == 'x':
            x_center = tick_position
            y_center = self.axes['x_axis'][1]  # Y position of x-axis
            
            # Search area below axis
            x1 = x_center - 30
            y1 = y_center - 5
            x2 = x_center + 30
            y2 = y_center + 35
            
        else:  # y-axis
            x_center = self.axes['y_axis'][0]  # X position of y-axis
            y_center = tick_position
            
            # Search area left of axis
            x1 = x_center - 60
            y1 = y_center - 15
            x2 = x_center - 5
            y2 = y_center + 15
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        # Extract ROI
        roi = self.prep.enhanced[y1:y2, x1:x2]
        
        # Try OCR
        try:
            text = pytesseract.image_to_string(
                roi, 
                config='--psm 7 -c tessedit_char_whitelist=0123456789.-+eE×'
            ).strip()
            
            if text:
                # Try to parse as number
                try:
                    text = text.replace('×', 'e')
                    value = float(text)
                    return {
                        'label': text,
                        'value': value,
                        'confidence': 0.8
                    }
                except:
                    return {
                        'label': text,
                        'value': None,
                        'confidence': 0.5
                    }
        except:
            pass
        
        return {'label': '?', 'value': None, 'confidence': 0}
    
    def _create_debug_visualization(self):
        """Create extreme debug visualization for tick detection"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TICK DETECTION - EXTREME DEBUG', fontsize=16, fontweight='bold')
        
        # 1. All tick marks visualized
        ax = axes[0, 0]
        img = self.image.copy()
        
        if self.axes['x_axis'] is not None:
            for tick in self.ticks['x_ticks']:
                x = tick['position']
                y = self.axes['x_axis'][1]
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.line(img, (x, y-10), (x, y+10), (0, 255, 0), 2)
                
                if tick['label']:
                    cv2.putText(img, tick['label'], (x-15, y+30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if self.axes['y_axis'] is not None:
            for tick in self.ticks['y_ticks']:
                x = self.axes['y_axis'][0]
                y = tick['position']
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                cv2.line(img, (x-10, y), (x+10, y), (255, 0, 0), 2)
                
                if tick['label']:
                    cv2.putText(img, tick['label'], (x-50, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('DETECTED TICKS & LABELS', fontweight='bold')
        ax.axis('off')
        
        # 2. OCR regions
        ax = axes[0, 1]
        img = self.image.copy()
        
        # Show OCR search regions
        if self.axes['x_axis'] is not None:
            for tick in self.ticks['x_ticks'][:5]:  # Show first 5 for clarity
                x = tick['position']
                y = self.axes['x_axis'][1]
                cv2.rectangle(img, (x-30, y), (x+30, y+30), (255, 255, 0), 1)
        
        if self.axes['y_axis'] is not None:
            for tick in self.ticks['y_ticks'][:5]:
                x = self.axes['y_axis'][0]
                y = tick['position']
                cv2.rectangle(img, (x-50, y-15), (x-5, y+15), (255, 255, 0), 1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('OCR SEARCH REGIONS', fontweight='bold')
        ax.axis('off')
        
        # 3. Tick statistics
        ax = axes[1, 0]
        ax.axis('off')
        
        x_labels = [t['label'] for t in self.ticks['x_ticks'] if t['label']]
        y_labels = [t['label'] for t in self.ticks['y_ticks'] if t['label']]
        
        info_text = f"""
        TICK DETECTION STATISTICS:
        
        X-AXIS:
        Ticks found: {len(self.ticks['x_ticks'])}
        Labels read: {len(x_labels)}
        First 5 labels: {x_labels[:5]}
        
        Y-AXIS:
        Ticks found: {len(self.ticks['y_ticks'])}
        Labels read: {len(y_labels)}
        First 5 labels: {y_labels[:5]}
        
        OCR Configuration:
        PSM: Single character/word
        Whitelist: 0-9, ., -, +, e, E, ×
        """
        ax.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=10,
               verticalalignment='center')
        
        # 4. Tick positions plot
        ax = axes[1, 1]
        
        if self.ticks['x_ticks']:
            x_positions = [t['position'] for t in self.ticks['x_ticks']]
            ax.scatter(x_positions, [0]*len(x_positions), c='green', s=100, alpha=0.6)
            
            # Add labels
            for tick in self.ticks['x_ticks']:
                if tick['label']:
                    ax.annotate(tick['label'], (tick['position'], 0),
                               xytext=(0, 10), textcoords='offset points',
                               fontsize=8, ha='center')
        
        if self.ticks['y_ticks']:
            y_positions = [t['position'] for t in self.ticks['y_ticks']]
            ax.scatter([0]*len(y_positions), y_positions, c='blue', s=100, alpha=0.6)
            
            for tick in self.ticks['y_ticks']:
                if tick['label']:
                    ax.annotate(tick['label'], (0, tick['position']),
                               xytext=(-30, 0), textcoords='offset points',
                               fontsize=8, ha='right')
        
        ax.set_title('TICK POSITIONS', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.prep.run_dir, '06_tick_detection_debug.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# LEGEND DETECTOR
# ============================================================================

class LegendDetector:
    """Detect and parse plot legends"""
    
    def __init__(self, preprocessor, axes):
        self.prep = preprocessor
        self.axes = axes
        self.config = preprocessor.config
        self.image = preprocessor.original.copy()
        self.height = preprocessor.height
        self.width = preprocessor.width
        self.legend_data = None
        
    def detect_legend(self):
        """Detect and parse legend"""
        # Search for legend in common positions
        candidates = self._find_legend_candidates()
        
        if candidates:
            best_candidate = self._select_best_candidate(candidates)
            self.legend_data = self._parse_legend(best_candidate)
        
        # Create debug visualization
        self._create_debug_visualization(candidates)
        
        return self.legend_data
    
    def _find_legend_candidates(self):
        """Find potential legend regions"""
        candidates = []
        pr = self.axes['plot_region']
        
        if pr is None:
            return candidates
        
        # Define search regions (top-right, top-left, etc.)
        search_regions = [
            # Top-right (inside plot)
            {
                'name': 'top_right',
                'x1': pr['x_max'] - 200,
                'y1': pr['y_min'],
                'x2': pr['x_max'],
                'y2': pr['y_min'] + 150
            },
            # Top-left (inside plot)
            {
                'name': 'top_left',
                'x1': pr['x_min'],
                'y1': pr['y_min'],
                'x2': pr['x_min'] + 200,
                'y2': pr['y_min'] + 150
            },
            # Outside right
            {
                'name': 'outside_right',
                'x1': pr['x_max'] + 10,
                'y1': pr['y_min'] + 50,
                'x2': min(self.width, pr['x_max'] + 200),
                'y2': min(self.height, pr['y_min'] + 250)
            }
        ]
        
        for region in search_regions:
            x1 = max(0, region['x1'])
            y1 = max(0, region['y1'])
            x2 = min(self.width, region['x2'])
            y2 = min(self.height, region['y2'])
            
            if x2 - x1 > 20 and y2 - y1 > 20:
                roi = self.prep.binary[y1:y2, x1:x2]
                
                # Check for text-like features
                contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Legend typically has multiple small contours
                if 2 < len(contours) < 50:
                    candidates.append({
                        'region': region,
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'num_contours': len(contours)
                    })
        
        return candidates
    
    def _select_best_candidate(self, candidates):
        """Select the most likely legend candidate"""
        # Simple heuristic: choose the one with most contours
        return max(candidates, key=lambda c: c['num_contours'])
    
    def _parse_legend(self, candidate):
        """Extract legend entries"""
        x, y, w, h = candidate['bbox']
        roi = self.prep.enhanced[y:y+h, x:x+w]
        
        # Try OCR on the entire region
        try:
            text = pytesseract.image_to_string(roi, config='--psm 6')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
        except:
            lines = []
        
        # Detect colored regions (legend markers)
        roi_color = self.image[y:y+h, x:x+w]
        
        # Find unique colors (potential legend markers)
        colors = self._extract_unique_colors(roi_color)
        
        return {
            'bbox': candidate['bbox'],
            'text_lines': lines,
            'colors': colors,
            'confidence': 0.7 if lines else 0.3
        }
    
    def _extract_unique_colors(self, image):
        """Extract unique colors from legend region"""
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove white/black pixels
        mask = (pixels[:, 0] > 20) & (pixels[:, 0] < 235)
        colored_pixels = pixels[mask]
        
        if len(colored_pixels) < 10:
            return []
        
        # Cluster colors
        clustering = DBSCAN(eps=10, min_samples=10).fit(colored_pixels)
        
        unique_colors = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster_pixels = colored_pixels[clustering.labels_ == label]
                avg_color = np.mean(cluster_pixels, axis=0)
                unique_colors.append(avg_color.astype(int).tolist())
        
        return unique_colors[:10]  # Limit to 10 colors
    
    def _create_debug_visualization(self, candidates):
        """Create debug visualization for legend detection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('LEGEND DETECTION - EXTREME DEBUG', fontsize=14, fontweight='bold')
        
        pr = self.axes['plot_region']
        
        # 1. Search regions
        ax = axes[0]
        img = self.image.copy()
        
        if pr:
            # Draw all search regions
            search_areas = [
                ('TOP-RIGHT', pr['x_max']-200, pr['y_min'], 200, 150, (0, 255, 0)),
                ('TOP-LEFT', pr['x_min'], pr['y_min'], 200, 150, (255, 0, 0)),
                ('RIGHT', pr['x_max']+10, pr['y_min']+50, 190, 200, (0, 0, 255))
            ]
            
            for name, x, y, w, h, color in search_areas:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, name, (x+5, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('LEGEND SEARCH REGIONS', fontweight='bold')
        ax.axis('off')
        
        # 2. Best candidate
        ax = axes[1]
        img = self.image.copy()
        
        if self.legend_data:
            x, y, w, h = self.legend_data['bbox']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(img, 'LEGEND FOUND!', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw legend text
            for i, line in enumerate(self.legend_data['text_lines']):
                cv2.putText(img, line[:30], (x+10, y+20*i+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif candidates:
            for cand in candidates[:3]:
                x, y, w, h = cand['bbox']
                cx, cy = x, y
                cv2.rectangle(img, (cx, cy), (cx+w, cy+h), (0, 0, 255), 2)
                cv2.putText(img, f"Candidate ({cand['num_contours']} objects)", 
                           (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('LEGEND CANDIDATES', fontweight='bold')
        ax.axis('off')
        
        # 3. Legend info
        ax = axes[2]
        ax.axis('off')
        
        if self.legend_data:
            x, y, w, h = self.legend_data['bbox']
            info_text = f"""
            LEGEND FOUND!
            
            Position: ({x}, {y})
            Size: {w}x{h}
            Text lines found: {len(self.legend_data['text_lines'])}
            Colors detected: {len(self.legend_data['colors'])}
            
            Legend Text:
            """
            for line in self.legend_data['text_lines'][:10]:
                info_text += f"  • {line}\n"
            
            info_text += f"""
            Confidence: {self.legend_data['confidence']:.2f}
            """
        else:
            num_candidates = len(candidates) if candidates else 0
            info_text = f"""
            NO LEGEND DETECTED
            
            Candidates found: {num_candidates}
            
            Tips:
            - Legend may be outside plot area
            - Check if plot has a legend
            - Try adjusting search regions
            """
        
        ax.text(0.05, 0.95, info_text, fontfamily='monospace', fontsize=9,
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.prep.run_dir, '07_legend_detection_debug.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# CURVE EXTRACTOR
# ============================================================================

class CurveExtractor:
    """Extract data curves from the plot"""
    
    def __init__(self, preprocessor, axes):
        self.prep = preprocessor
        self.axes = axes
        self.config = preprocessor.config
        self.image = preprocessor.original.copy()
        self.height = preprocessor.height
        self.width = preprocessor.width
        self.curves = []
        
    def extract_curves(self):
        """Extract all data curves from the plot area"""
        pr = self.axes['plot_region']
        
        if pr is None:
            return self.curves
        
        # Extract plot area
        plot_area = self.image[
            pr['y_min']:pr['y_max'],
            pr['x_min']:pr['x_max']
        ]
        
        if plot_area.size == 0:
            return self.curves
        
        # Detect curves by color
        color_masks = self._detect_curve_colors(plot_area)
        
        # Extract points for each color
        for i, (color, mask) in enumerate(color_masks.items()):
            curve_points = self._extract_curve_points(mask)
            
            if len(curve_points) > self.config.MIN_CURVE_POINTS:
                # Convert to plot coordinates
                curve_points[:, 0] += pr['x_min']  # Add x offset
                curve_points[:, 1] += pr['y_min']  # Add y offset
                
                self.curves.append({
                    'id': i,
                    'color': list(color),
                    'points': curve_points,
                    'num_points': len(curve_points)
                })
        
        # Create debug visualization
        self._create_debug_visualization()
        
        return self.curves
    
    def _detect_curve_colors(self, plot_area):
        """Detect different colored curves in the plot area"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(plot_area, cv2.COLOR_BGR2HSV)
        
        # Remove white background and grid
        # White has low saturation
        saturation = hsv[:, :, 1]
        _, mask_saturated = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
        
        # Remove very dark pixels (potential text)
        value = hsv[:, :, 2]
        _, mask_not_black = cv2.threshold(value, 30, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(mask_saturated, mask_not_black)
        
        # Find unique colors in remaining pixels
        colors = self._find_clusters(hsv, combined_mask)
        
        # Create color masks
        color_masks = {}
        for color in colors:
            mask = self._create_color_mask(hsv, color)
            color_masks[tuple(color)] = mask
        
        return color_masks
    
    def _find_clusters(self, hsv, mask):
        """Find dominant color clusters"""
        pixels = hsv[mask > 0]
        
        if len(pixels) < 100:
            return []
        
        # Cluster in HSV space
        clustering = DBSCAN(eps=20, min_samples=50).fit(pixels)
        
        colors = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster = pixels[clustering.labels_ == label]
                avg_color = np.median(cluster, axis=0)
                colors.append(avg_color)
        
        # Filter out similar colors
        unique_colors = []
        for color in colors:
            is_unique = True
            for existing in unique_colors:
                if np.linalg.norm(color - existing) < 30:
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(color)
        
        return unique_colors[:5]  # Limit to 5 curves
    
    def _create_color_mask(self, hsv, color, threshold=30):
        """Create mask for a specific color"""
        lower = np.array([
            max(0, color[0] - threshold),
            max(0, color[1] - threshold),
            max(0, color[2] - threshold)
        ])
        upper = np.array([
            min(180, color[0] + threshold),
            min(255, color[1] + threshold),
            min(255, color[2] + threshold)
        ])
        
        return cv2.inRange(hsv, lower, upper)
    
    def _extract_curve_points(self, mask):
        """Extract curve points from a color mask"""
        if mask.size == 0 or np.sum(mask) == 0:
            return np.array([])
        
        # Find non-zero points
        points = np.column_stack(np.where(mask > 0))
        
        if len(points) > 0:
            # Sort points by x coordinate
            points = points[points[:, 1].argsort()]
            
            # Remove duplicate x values (keep first occurrence)
            _, unique_indices = np.unique(points[:, 1], return_index=True)
            points = points[sorted(unique_indices)]
            
            # Swap to (x, y) format
            points = points[:, [1, 0]]
        
        return points
    
    def _create_debug_visualization(self):
        """Create extreme debug visualization for curves"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('CURVE EXTRACTION - EXTREME DEBUG', fontsize=16, fontweight='bold')
        
        pr = self.axes['plot_region']
        
        # 1. Plot area with detected curves
        ax = axes[0, 0]
        img = self.image.copy()
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
        for i, curve in enumerate(self.curves):
            color = colors[i % len(colors)]
            for point in curve['points'][::5]:  # Plot every 5th point
                cv2.circle(img, (int(point[0]), int(point[1])), 2, color, -1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'DETECTED CURVES ({len(self.curves)})', fontweight='bold')
        ax.axis('off')
        
        # 2. Color masks
        ax = axes[0, 1]
        if self.curves and pr:
            # Create composite of all masks
            composite = np.zeros((pr['y_max']-pr['y_min'], pr['x_max']-pr['x_min'], 3), dtype=np.uint8)
            
            for i, curve in enumerate(self.curves):
                color = colors[i % len(colors)]
                mask = np.zeros_like(composite)
                for point in curve['points']:
                    px = int(point[0] - pr['x_min'])
                    py = int(point[1] - pr['y_min'])
                    if 0 <= py < composite.shape[0] and 0 <= px < composite.shape[1]:
                        mask[py, px] = color
                composite = cv2.add(composite, mask)
            
            ax.imshow(composite)
        ax.set_title('COLOR MASKS', fontweight='bold')
        ax.axis('off')
        
        # 3. Curve skeletons
        ax = axes[0, 2]
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i, curve in enumerate(self.curves):
            color = colors[i % len(colors)]
            for point in curve['points']:
                cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
        
        ax.imshow(img)
        ax.set_title('CURVE SKELETONS', fontweight='bold')
        ax.axis('off')
        
        # 4. Point distribution
        ax = axes[1, 0]
        for i, curve in enumerate(self.curves):
            color = [c/255 for c in colors[i % len(colors)]]
            ax.scatter(curve['points'][::10, 0], curve['points'][::10, 1], 
                      color=color, s=1, alpha=0.5, label=f'Curve {i+1}')
        
        ax.invert_yaxis()
        ax.set_title('POINT DISTRIBUTION', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(self.curves) <= 5:
            ax.legend()
        
        # 5. Curve statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = f"""
        CURVE EXTRACTION STATISTICS:
        
        Total curves detected: {len(self.curves)}
        
        """
        for i, curve in enumerate(self.curves):
            info_text += f"""
        Curve {i+1}:
        - Points: {curve['num_points']}
        - X range: {curve['points'][:, 0].min():.0f} to {curve['points'][:, 0].max():.0f}
        - Y range: {curve['points'][:, 1].min():.0f} to {curve['points'][:, 1].max():.0f}
        """
        
        ax.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=9,
               verticalalignment='center')
        
        # 6. Point density
        ax = axes[1, 2]
        for i, curve in enumerate(self.curves):
            # Calculate point density
            x_diff = np.diff(curve['points'][:, 0])
            if len(x_diff[x_diff > 0]) > 0:
                ax.hist(x_diff[x_diff > 0], bins=30, alpha=0.5, 
                       label=f'Curve {i+1}', color=[c/255 for c in colors[i % len(colors)]])
        
        ax.set_xlabel('X-distance between points (px)')
        ax.set_ylabel('Frequency')
        ax.set_title('POINT DENSITY DISTRIBUTION', fontweight='bold')
        if len(self.curves) <= 5:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.prep.run_dir, '08_curve_extraction_debug.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class AutoPlotDigitizer:
    """Main class orchestrating the automatic extraction"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.preprocessor = ImagePreprocessor(self.config)
        
    def extract(self, image_path):
        """Complete extraction pipeline"""
        print(f"\n{'='*60}")
        print(f"AUTO PLOT DIGITIZER - STARTING EXTRACTION")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Debug: {self.config.DEBUG_LEVEL.upper()}")
        
        # Step 1: Preprocess
        print("\n[1/5] Preprocessing image...")
        self.preprocessor.preprocess(image_path)
        
        # Step 2: Detect axes
        print("[2/5] Detecting axes...")
        axis_detector = AxisDetector(self.preprocessor)
        axes = axis_detector.detect()
        
        # Step 3: Detect ticks
        print("[3/5] Detecting ticks and labels...")
        tick_detector = TickDetector(self.preprocessor, axes)
        ticks = tick_detector.detect_ticks()
        
        # Step 4: Detect legend
        print("[4/5] Detecting legend...")
        legend_detector = LegendDetector(self.preprocessor, axes)
        legend = legend_detector.detect_legend()
        
        # Step 5: Extract curves
        print("[5/5] Extracting curves...")
        curve_extractor = CurveExtractor(self.preprocessor, axes)
        curves = curve_extractor.extract_curves()
        
        # Compile results
        results = {
            'image': image_path,
            'axes': axes,
            'ticks': ticks,
            'legend': legend,
            'curves': curves,
            'debug_dir': self.preprocessor.run_dir
        }
        
        # Save results
        self._save_results(results)
        
        # Create final visualization
        self._create_final_visualization(results)
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE!")
        print(f"Debug outputs: {self.preprocessor.run_dir}")
        print(f"{'='*60}\n")
        
        return results
    
    def _save_results(self, results):
        """Save extraction results to JSON"""
        # Prepare serializable data
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable = {
            'image': results['image'],
            'axes': {
                'x_axis': results['axes']['x_axis'] if results['axes']['x_axis'] is not None else None,
                'y_axis': results['axes']['y_axis'] if results['axes']['y_axis'] is not None else None,
                'plot_region': results['axes']['plot_region']
            },
            'ticks': {
                'x_ticks': results['ticks']['x_ticks'],
                'y_ticks': results['ticks']['y_ticks']
            },
            'legend': results['legend'],
            'curves': [
                {
                    'id': c['id'],
                    'color': c['color'],
                    'num_points': c['num_points'],
                    'points': c['points'].tolist() if isinstance(c['points'], np.ndarray) else c['points']
                }
                for c in results['curves']
            ]
        }
        
        # Convert numpy types
        serializable = convert_to_serializable(serializable)
        
        json_path = os.path.join(results['debug_dir'], 'extraction_results.json')
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to: {json_path}")
    
    def _create_final_visualization(self, results):
        """Create comprehensive final visualization"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('AUTO PLOT DIGITIZER - FINAL RESULTS', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Original with annotations
        ax = fig.add_subplot(gs[0, :])
        img = cv2.imread(results['image'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw plot region
        if results['axes']['plot_region']:
            pr = results['axes']['plot_region']
            rect = Rectangle((pr['x_min'], pr['y_min']),
                           pr['x_max'] - pr['x_min'],
                           pr['y_max'] - pr['y_min'],
                           fill=False, edgecolor='yellow', linewidth=2)
            ax.add_patch(rect)
        
        # Draw axes
        if results['axes']['x_axis'] is not None:
            x1, y1, x2, y2 = results['axes']['x_axis']
            ax.plot([x1, x2], [y1, y2], 'c-', linewidth=2, label='X-Axis')
        
        if results['axes']['y_axis'] is not None:
            x1, y1, x2, y2 = results['axes']['y_axis']
            ax.plot([x1, x2], [y1, y2], 'm-', linewidth=2, label='Y-Axis')
        
        # Draw curves
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, curve in enumerate(results['curves']):
            color = colors[i % len(colors)]
            points = curve['points']
            ax.scatter(points[::20, 0], points[::20, 1], 
                      c=color, s=2, alpha=0.6, label=f'Curve {i+1}')
        
        ax.imshow(img_rgb)
        ax.set_title('ORIGINAL IMAGE WITH DETECTIONS', fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # 2-4. Individual curves
        for i, curve in enumerate(results['curves'][:3]):
            ax = fig.add_subplot(gs[1, i])
            points = curve['points']
            ax.scatter(points[::5, 0], points[::5, 1], s=1, alpha=0.5)
            
            ax.set_xlabel('X position (pixels)')
            ax.set_ylabel('Y position (pixels)')
            
            ax.set_title(f'CURVE {i+1} DETAIL', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        
        # Fill remaining spots if less than 3 curves
        for i in range(len(results['curves']), 3):
            ax = fig.add_subplot(gs[1, i])
            ax.text(0.5, 0.5, f'NO CURVE {i+1}', ha='center', va='center')
            ax.set_title(f'CURVE {i+1} DETAIL', fontweight='bold')
            ax.axis('off')
        
        # 5. Tick analysis
        ax = fig.add_subplot(gs[2, 0])
        
        if results['ticks']['x_ticks']:
            x_pos = [t['position'] for t in results['ticks']['x_ticks']]
            x_val = [t.get('value', i) for i, t in enumerate(results['ticks']['x_ticks'])]
            ax.scatter(x_pos, [0]*len(x_pos), c='green', s=50)
            for pos, val in zip(x_pos, x_val):
                if val is not None:
                    ax.annotate(f'{val:.1f}', (pos, 0), 
                              xytext=(0, 15), textcoords='offset points',
                              fontsize=7, ha='center')
        
        if results['ticks']['y_ticks']:
            y_pos = [t['position'] for t in results['ticks']['y_ticks']]
            y_val = [t.get('value', i) for i, t in enumerate(results['ticks']['y_ticks'])]
            ax.scatter([0]*len(y_pos), y_pos, c='blue', s=50)
            for pos, val in zip(y_pos, y_val):
                if val is not None:
                    ax.annotate(f'{val:.1f}', (0, pos),
                              xytext=(-40, 0), textcoords='offset points',
                              fontsize=7, ha='right')
        
        ax.set_title('TICK ANALYSIS', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 6. Statistics
        ax = fig.add_subplot(gs[2, 1])
        ax.axis('off')
        
        stats = f"""
        EXTRACTION SUMMARY:
        
        ✓ Axes detected: {bool(results['axes']['x_axis'] or results['axes']['y_axis'])}
        ✓ Plot region: {bool(results['axes']['plot_region'])}
        ✓ X-ticks found: {len(results['ticks']['x_ticks'])}
        ✓ Y-ticks found: {len(results['ticks']['y_ticks'])}
        ✓ Curves extracted: {len(results['curves'])}
        ✓ Legend detected: {results['legend'] is not None}
        
        Total points extracted: {sum(c['num_points'] for c in results['curves'])}
        
        Debug directory:
        {results['debug_dir']}
        """
        
        ax.text(0.1, 0.5, stats, fontfamily='monospace', fontsize=10,
               verticalalignment='center', transform=ax.transAxes)
        
        # 7. Legend
        ax = fig.add_subplot(gs[2, 2])
        if results['legend']:
            legend_text = "LEGEND DETECTED:\n\n"
            for line in results['legend']['text_lines'][:8]:
                legend_text += f"• {line}\n"
            ax.text(0.1, 0.9, legend_text, fontfamily='monospace', fontsize=8,
                   verticalalignment='top', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'NO LEGEND DETECTED',
                   ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.set_title('LEGEND INFO', fontweight='bold')
        
        # Save
        final_path = os.path.join(results['debug_dir'], '09_FINAL_RESULTS.png')
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Final visualization saved to: {final_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'plot.png'  # Default image name
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        print("Usage: python auto_digitizer.py <image_path>")
        print("Make sure to place your plot image in the current directory")
        sys.exit(1)
    
    # Create extractor with config
    config = Config()
    config.DEBUG_LEVEL = 'extreme'
    
    extractor = AutoPlotDigitizer(config)
    results = extractor.extract(image_path)
    
    print("\nExtraction complete! Check the debug_outputs directory for:")
    print("  00_original.png - Original image")
    print("  01_denoised.png - Denoised image")
    print("  02_enhanced.png - Contrast enhanced")
    print("  03_binary.png - Binary image")
    print("  04_edges.png - Edge detection")
    print("  05_axis_detection_debug.png - Axis detection analysis")
    print("  06_tick_detection_debug.png - Tick detection analysis")
    print("  07_legend_detection_debug.png - Legend detection analysis")
    print("  08_curve_extraction_debug.png - Curve extraction analysis")
    print("  09_FINAL_RESULTS.png - Complete extraction results")
    print("  extraction_results.json - JSON with all extracted data")

if __name__ == "__main__":
    main()