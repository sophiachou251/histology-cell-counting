# Configuration for cell counting analysis

# Image processing parameters
BLUR_KERNEL = 5
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150
MORPHOLOGICAL_KERNEL_SIZE = 3

# Cell detection parameters
MIN_CELL_AREA = 20  # Minimum area in pixels to consider as a cell
MAX_CELL_AREA = 800  # Maximum area in pixels to consider as a cell
CIRCULARITY_THRESHOLD = 0.4  # Minimum circularity (0-1, where 1 is perfect circle)
SOLIDITY_THRESHOLD = 0.7  # Minimum solidity (0-1, where 1 is solid)

# Color-based detection thresholds (HSV ranges for different cell types)
RED_BLOOD_CELLS = {
    'color': 'red',
    'hsv_lower': (0, 100, 100),
    'hsv_upper': (10, 255, 255),
    'area_range': (30, 300)
}

WHITE_BLOOD_CELLS = {
    'color': 'white',
    'hsv_lower': (0, 0, 200),
    'hsv_upper': (180, 30, 255),
    'area_range': (50, 600)
}

PLATELETS = {
    'color': 'blue',
    'hsv_lower': (100, 100, 50),
    'hsv_upper': (130, 255, 200),
    'area_range': (5, 100)
}

# Output settings
OUTPUT_DIR = 'results'
SAVE_VISUALIZATIONS = True
SAVE_CSV_RESULTS = True

# Analysis settings
BATCH_PROCESS = True  # Process all images in JPEGImages folder
VERBOSE = True  # Print detailed output during processing