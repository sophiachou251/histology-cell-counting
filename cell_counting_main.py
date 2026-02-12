import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from cell_counting_rules import CellCountingRules
from config import *

class CellCountingAnalyzer:
    """
    Main class for analyzing blood cell images and counting different cell types
    """
    
    def __init__(self, image_dir='JPEGImages', annotation_dir='Annotations', output_dir=OUTPUT_DIR):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_image(self, image_path):
        """Load an image from file"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        return image
    
    def preprocess_image(self, image):
        """Preprocess image for cell detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_cells(self, image):
        """Detect cell contours in image"""
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Apply edge detection
        edges = cv2.Canny(processed, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def count_cells(self, image):
        """Count different cell types in image"""
        contours = self.detect_cells(image)
        
        # Filter overlapping cells
        contours = CellCountingRules.filter_overlapping_cells(contours)
        
        # Detect different cell types
        red_blood_cells = CellCountingRules.detect_red_blood_cells(image, contours)
        white_blood_cells = CellCountingRules.detect_white_blood_cells(image, contours)
        platelets = CellCountingRules.detect_platelets(image, contours)
        
        return {
            'red_blood_cells': len(red_blood_cells),
            'white_blood_cells': len(white_blood_cells),
            'platelets': len(platelets),
            'total_cells': len(red_blood_cells) + len(white_blood_cells) + len(platelets),
            'contours': contours,
            'red_contours': red_blood_cells,
            'white_contours': white_blood_cells,
            'platelet_contours': platelets
        }
    
    def visualize_results(self, image, results, output_path):
        """Visualize cell detection results"""
        vis_image = image.copy()
        
        # Draw red blood cells in red
        for contour in results['red_contours']:
            cv2.drawContours(vis_image, [contour], 0, (0, 0, 255), 2)
        
        # Draw white blood cells in green
        for contour in results['white_contours']:
            cv2.drawContours(vis_image, [contour], 0, (0, 255, 0), 2)
        
        # Draw platelets in blue
        for contour in results['platelet_contours']:
            cv2.drawContours(vis_image, [contour], 0, (255, 0, 0), 2)
        
        # Add text with counts
        text = f"RBC: {results['red_blood_cells']}, WBC: {results['white_blood_cells']}, PLT: {results['platelets']}"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        return vis_image
    
    def process_image(self, image_path):
        """Process single image and return results"""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        results = self.count_cells(image)
        
        # Store results
        image_name = os.path.basename(image_path)
        result_data = {
            'image': image_name,
            'red_blood_cells': results['red_blood_cells'],
            'white_blood_cells': results['white_blood_cells'],
            'platelets': results['platelets'],
            'total_cells': results['total_cells']
        }
        self.results.append(result_data)
        
        if VERBOSE:
            print(f"Processed {image_name}: RBC={results['red_blood_cells']}, WBC={results['white_blood_cells']}, PLT={results['platelets']}")
        
        # Visualize and save
        if SAVE_VISUALIZATIONS:
            output_path = os.path.join(self.output_dir, f'annotated_{image_name}')
            self.visualize_results(image, results, output_path)
        
        return result_data
    
    def batch_process(self):
        """Process all images in the JPEGImages directory"""
        if not os.path.exists(self.image_dir):
            print(f"Error: Image directory '{self.image_dir}' not found")
            return
        
        image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images to process")
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(self.image_dir, image_file)
            print(f"[{i}/{len(image_files)}] Processing {image_file}...")
            self.process_image(image_path)
        
        print(f"Processing complete! Processed {len(image_files)} images")
    
    def save_results(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        output_path = os.path.join(self.output_dir, f'cell_counts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total images processed: {len(df)}")
        print(f"\nRed Blood Cells (RBC):")
        print(f"  Mean: {df['red_blood_cells'].mean():.1f}")
        print(f"  Median: {df['red_blood_cells'].median():.1f}")
        print(f"  Std Dev: {df['red_blood_cells'].std():.1f}")
        
        print(f"\nWhite Blood Cells (WBC):")
        print(f"  Mean: {df['white_blood_cells'].mean():.1f}")
        print(f"  Median: {df['white_blood_cells'].median():.1f}")
        print(f"  Std Dev: {df['white_blood_cells'].std():.1f}")
        
        print(f"\nPlatelets (PLT):")
        print(f"  Mean: {df['platelets'].mean():.1f}")
        print(f"  Median: {df['platelets'].median():.1f}")
        print(f"  Std Dev: {df['platelets'].std():.1f}")
        
        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    """Main execution function"""
    print("=== Blood Cell Counting Analysis ===\n")
    
    # Create analyzer
    analyzer = CellCountingAnalyzer()
    
    # Process all images
    if BATCH_PROCESS:
        analyzer.batch_process()
    
    # Save results
    if SAVE_CSV_RESULTS:
        analyzer.save_results()


if __name__ == "__main__":
    main()