import numpy as np
import json
from typing import List, Tuple, Dict, Any

class Camera:
    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray, 
                 fov: float, aspect_ratio: float, near: float, far: float):
        self.position = position
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        
        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
    
    def _compute_view_matrix(self) -> np.ndarray:
        """Compute the view matrix using camera position, target, and up vector"""
        forward = (self.target - self.position) / np.linalg.norm(self.target - self.position)
        right = np.cross(forward, self.up) / np.linalg.norm(np.cross(forward, self.up))
        up = np.cross(right, forward)
        
        view = np.eye(4)
        view[:3, 0] = right
        view[:3, 1] = up
        view[:3, 2] = -forward
        view[:3, 3] = -np.dot(view[:3, :3], self.position)
        
        return view
    
    def _compute_projection_matrix(self) -> np.ndarray:
        """Compute the perspective projection matrix"""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        projection = np.zeros((4, 4))
        
        projection[0, 0] = f / self.aspect_ratio
        projection[1, 1] = f
        projection[2, 2] = (self.far + self.near) / (self.near - self.far)
        projection[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        projection[3, 2] = -1.0
        
        return projection
    
    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project a 3D point to 2D screen space"""
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point, 1.0)
        
        # Apply view and projection transformations
        view_proj = np.dot(self.projection_matrix, self.view_matrix)
        projected = np.dot(view_proj, point_homogeneous)
        
        # Perspective divide
        if projected[3] != 0:
            projected = projected / projected[3]
        
        return projected[:2]

def create_camera_ring(center: np.ndarray, radius: float, num_views: int, 
                      height: float = 1.0) -> List[Camera]:
    """
    Create a ring of cameras around a center point
    
    Args:
        center: Center point of the scene
        radius: Radius of the camera ring
        num_views: Number of cameras to create
        height: Height offset of cameras from the center
        
    Returns:
        List of Camera instances
    """
    cameras = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        position = np.array([x, center[1] + height, z])
        
        camera = Camera(
            position=position,
            target=center,
            up=np.array([0, 1, 0]),
            fov=60.0,
            aspect_ratio=1.0,
            near=0.1,
            far=100.0
        )
        cameras.append(camera)
    
    return cameras

def generate_mcp_context(objects: List[Any], image_paths: List[str], 
                        camera_positions: List[np.ndarray]) -> Dict[str, Any]:
    """
    Generate a Model Context Protocol (MCP) JSON context for LLM labeling
    
    Args:
        objects: List of detected 3D objects
        image_paths: List of paths to the generated 2D images
        camera_positions: List of camera positions used for each view
        
    Returns:
        Dictionary containing MCP context data
    """
    context = {
        "scene": {
            "objects": [],
            "views": []
        },
        "metadata": {
            "version": "1.0",
            "generator": "3DETR-MCP",
            "date_created": "",
            "num_objects": len(objects),
            "num_views": len(image_paths)
        }
    }
    
    # Add timestamp
    import datetime
    context["metadata"]["date_created"] = datetime.datetime.now().isoformat()
    
    # Add objects
    for i, obj in enumerate(objects):
        object_data = {
            "id": i,
            "class_name": obj.class_name if hasattr(obj, "class_name") else "unknown",
            "confidence": float(obj.confidence) if hasattr(obj, "confidence") else 0.0,
            "bbox_3d": obj.bbox.tolist() if hasattr(obj, "bbox") else [],
            "position": [float(obj.bbox[0]), float(obj.bbox[1]), float(obj.bbox[2])] 
                       if hasattr(obj, "bbox") else [0, 0, 0]
        }
        context["scene"]["objects"].append(object_data)
    
    # Add views
    for i, (image_path, camera_pos) in enumerate(zip(image_paths, camera_positions)):
        view_data = {
            "id": i,
            "image_path": image_path,
            "camera_position": camera_pos.tolist(),
            "angle": (i * 360 / len(image_paths))
        }
        context["scene"]["views"].append(view_data)
    
    return context

def save_mcp_context(context: Dict[str, Any], output_path: str) -> None:
    """
    Save MCP context to a JSON file
    
    Args:
        context: Dictionary containing MCP context data
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(context, f, indent=2)
