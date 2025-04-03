import json
import numpy as np
from typing import Dict, List, Tuple, Optional

class Object3D:
    def __init__(self, class_name: str, bbox: np.ndarray, confidence: float, 
                object_id: Optional[int] = None, attributes: Optional[Dict] = None):
        self.class_name = class_name
        self.bbox = bbox  # [x, y, z, width, height, depth]
        self.confidence = confidence
        self.object_id = object_id
        self.attributes = attributes or {}
        
    def to_dict(self) -> Dict:
        """Convert object to dictionary representation for serialization"""
        return {
            "class_name": self.class_name,
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            "confidence": float(self.confidence),
            "object_id": self.object_id,
            "attributes": self.attributes,
            "center": [float(self.bbox[0]), float(self.bbox[1]), float(self.bbox[2])]
        }

def parse_3detr_output(output_path: str) -> List[Object3D]:
    """
    Parse 3DETR model output and extract object information
    
    Args:
        output_path: Path to the 3DETR output JSON file
        
    Returns:
        List of Object3D instances containing object information
    """
    with open(output_path, 'r') as f:
        output_data = json.load(f)
    
    objects = []
    for i, obj in enumerate(output_data['objects']):
        class_name = obj['class_name']
        bbox = np.array(obj['bbox'])  # [x, y, z, width, height, depth]
        confidence = obj['confidence']
        
        # Extract additional attributes if available
        attributes = {}
        for key, value in obj.items():
            if key not in ['class_name', 'bbox', 'confidence']:
                attributes[key] = value
        
        objects.append(Object3D(class_name, bbox, confidence, object_id=i, attributes=attributes))
    
    return objects

def get_object_vertices(obj: Object3D) -> np.ndarray:
    """
    Get the 8 vertices of a 3D bounding box
    
    Args:
        obj: Object3D instance
        
    Returns:
        numpy array of shape (8, 3) containing the vertices
    """
    x, y, z, w, h, d = obj.bbox
    vertices = np.array([
        [x - w/2, y - h/2, z - d/2],
        [x + w/2, y - h/2, z - d/2],
        [x + w/2, y + h/2, z - d/2],
        [x - w/2, y + h/2, z - d/2],
        [x - w/2, y - h/2, z + d/2],
        [x + w/2, y - h/2, z + d/2],
        [x + w/2, y + h/2, z + d/2],
        [x - w/2, y + h/2, z + d/2]
    ])
    return vertices

def calculate_object_relations(objects: List[Object3D]) -> Dict:
    """
    Calculate spatial relationships between objects
    
    Args:
        objects: List of Object3D instances
        
    Returns:
        Dictionary containing spatial relationship information
    """
    relations = {}
    
    for i, obj1 in enumerate(objects):
        relations[i] = []
        x1, y1, z1 = obj1.bbox[0], obj1.bbox[1], obj1.bbox[2]
        
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
                
            x2, y2, z2 = obj2.bbox[0], obj2.bbox[1], obj2.bbox[2]
            
            # Calculate Euclidean distance
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Determine direction (simplified)
            dx, dy, dz = x2-x1, y2-y1, z2-z1
            
            # Determine primary direction
            abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
            max_component = max(abs_dx, abs_dy, abs_dz)
            
            if max_component == abs_dx:
                direction = "right" if dx > 0 else "left"
            elif max_component == abs_dy:
                direction = "above" if dy > 0 else "below"
            else:
                direction = "front" if dz > 0 else "behind"
            
            relations[i].append({
                "object_id": j,
                "distance": float(distance),
                "direction": direction
            })
    
    return relations
