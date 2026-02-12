import cv2
import numpy as np
from config import *

class CellCountingRules:
    """
    Rules and heuristics for identifying and counting different cell types
    """
    
    @staticmethod
    def detect_red_blood_cells(image, contours):
        """Detect red blood cells based on morphological features"""
        red_cells = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints
            if area < RED_BLOOD_CELLS['area_range'][0] or area > RED_BLOOD_CELLS['area_range'][1]:
                continue
            
            # Calculate morphological features
            circularity = CellCountingRules.calculate_circularity(contour)
            solidity = CellCountingRules.calculate_solidity(contour)
            
            # Red blood cells are typically circular with good solidity
            if circularity > CIRCULARITY_THRESHOLD and solidity > SOLIDITY_THRESHOLD:
                red_cells.append(contour)
        
        return red_cells
    
    @staticmethod
    def detect_white_blood_cells(image, contours):
        """Detect white blood cells based on size and morphology"""
        white_cells = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # White blood cells are larger than red blood cells
            if area < WHITE_BLOOD_CELLS['area_range'][0] or area > WHITE_BLOOD_CELLS['area_range'][1]:
                continue
            
            solidity = CellCountingRules.calculate_solidity(contour)
            
            # White blood cells have good solidity
            if solidity > SOLIDITY_THRESHOLD * 0.9:
                white_cells.append(contour)
        
        return white_cells
    
    @staticmethod
    def detect_platelets(image, contours):
        """Detect platelets - smallest cell type"""
        platelets = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Platelets are very small
            if area < PLATELETS['area_range'][0] or area > PLATELETS['area_range'][1]:
                continue
            
            platelets.append(contour)
        return platelets
    
    @staticmethod
    def calculate_circularity(contour):
        """
        Calculate circularity of a contour
        Circularity = 4π * Area / Perimeter²
        Perfect circle = 1, other shapes < 1
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity
    
    @staticmethod
    def calculate_solidity(contour):
        """
        Calculate solidity of a contour
        Solidity = Area / Convex Hull Area
        Solid shape = 1, irregular shape < 1
        """
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0
        
        solidity = area / hull_area
        return solidity
    
    @staticmethod
    def calculate_eccentricity(contour):
        """
        Calculate eccentricity (how elongated a shape is)
        Circle = 0, elongated = closer to 1
        """
        if len(contour) < 5:
            return 0
        
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        
        if major_axis == 0:
            return 0
        
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return eccentricity
    
    @staticmethod
    def filter_overlapping_cells(contours, overlap_threshold=0.3):
        """
        Remove overlapping contours to avoid double-counting
        """
        if not contours:
            return contours
        
        filtered = []
        used = set()
        
        for i, contour1 in enumerate(contours):
            if i in used:
                continue
            
            filtered.append(contour1)
            
            for j, contour2 in enumerate(contours[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Calculate overlap between contours
                overlap = CellCountingRules.calculate_contour_overlap(contour1, contour2)
                
                if overlap > overlap_threshold:
                    used.add(j)
        
        return filtered
    
    @staticmethod
    def calculate_contour_overlap(contour1, contour2):
        """Calculate overlap ratio between two contours"""
        mask1 = np.zeros((500, 500), dtype=np.uint8)
        mask2 = np.zeros((500, 500), dtype=np.uint8)
        
        cv2.drawContours(mask1, [contour1], 0, 255, -1)
        cv2.drawContours(mask2, [contour2], 0, 255, -1)
        
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        
        intersection_area = np.sum(intersection > 0)
        union_area = np.sum(union > 0)
        
        if union_area == 0:
            return 0
        
        overlap = intersection_area / union_area
        return overlap
