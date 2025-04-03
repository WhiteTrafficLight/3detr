import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Any
from parse_3detr_output import Object3D, get_object_vertices, calculate_object_relations
from mcp_builder import Camera, create_camera_ring, generate_mcp_context, save_mcp_context

def create_blank_image(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a blank white image"""
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def draw_bbox_2d(image: np.ndarray, vertices_2d: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    """
    Draw a 2D bounding box on the image
    
    Args:
        image: Target image
        vertices_2d: 2D vertices of the bounding box
        color: RGB color tuple
    """
    # Convert vertices to integer coordinates
    vertices = vertices_2d.astype(int)
    
    # Draw edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for edge in edges:
        cv2.line(image, tuple(vertices[edge[0]]), tuple(vertices[edge[1]]), color, 2)

def generate_individual_object_image(obj: Object3D, camera: Camera, image_size: int = 512) -> np.ndarray:
    """
    Generate a 2D image of a single 3D object from a specific viewpoint
    
    Args:
        obj: Object3D instance
        camera: Camera instance for the viewpoint
        image_size: Size of the output image
        
    Returns:
        Image as numpy array
    """
    # Create blank image
    image = create_blank_image(width=image_size, height=image_size)
    
    # Get object vertices
    vertices_3d = get_object_vertices(obj)
    
    # Project vertices to 2D
    vertices_2d = np.array([camera.project_point(v) for v in vertices_3d])
    
    # Scale to image coordinates
    vertices_2d = (vertices_2d + 1) * (image_size / 2)
    
    # Draw bounding box
    color = (0, 0, 255)  # Red color
    draw_bbox_2d(image, vertices_2d, color)
    
    # Add class label
    label = f"{obj.class_name} ({obj.confidence:.2f})"
    cv2.putText(image, label, tuple(vertices_2d[0].astype(int)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def generate_object_images(objects: List[Object3D], output_dir: str, num_views: int = 8, 
                         generate_individual: bool = True, image_size: int = 512) -> Dict[str, Any]:
    """
    Generate 2D images of 3D objects from multiple viewpoints and create MCP context
    
    Args:
        objects: List of Object3D instances
        output_dir: Directory to save the generated images
        num_views: Number of viewpoints to render
        generate_individual: Whether to generate individual images for each object
        image_size: Size of the generated images
        
    Returns:
        MCP context dictionary
    """
    # Calculate scene center
    all_vertices = np.vstack([get_object_vertices(obj) for obj in objects])
    scene_center = np.mean(all_vertices, axis=0)
    
    # Create cameras
    cameras = create_camera_ring(
        center=scene_center,
        radius=5.0,  # Adjust based on scene size
        num_views=num_views
    )
    
    # Keep track of image paths and camera positions for MCP context
    image_paths = []
    camera_positions = []
    
    # Calculate object relations
    object_relations = calculate_object_relations(objects)
    
    # Create directory for individual object images if needed
    if generate_individual:
        individual_dir = os.path.join(output_dir, "individual_objects")
        os.makedirs(individual_dir, exist_ok=True)
    
    # Generate images for each viewpoint
    for view_idx, camera in enumerate(cameras):
        # Create scene image (with all objects)
        scene_image = create_blank_image(width=image_size, height=image_size)
        
        for obj_idx, obj in enumerate(objects):
            # Get object vertices
            vertices_3d = get_object_vertices(obj)
            
            # Project vertices to 2D
            vertices_2d = np.array([camera.project_point(v) for v in vertices_3d])
            
            # Convert to pixel coordinates
            vertices_2d = (vertices_2d + 1) * (image_size / 2)
            
            # Draw bounding box in scene image
            color = (0, 0, 255)  # Red color
            draw_bbox_2d(scene_image, vertices_2d, color)
            
            # Add class label
            label = f"{obj.class_name} ({obj.confidence:.2f})"
            cv2.putText(scene_image, label, tuple(vertices_2d[0].astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Generate individual object image if requested
            if generate_individual:
                individual_image = generate_individual_object_image(obj, camera, image_size)
                individual_path = os.path.join(individual_dir, f"obj_{obj_idx:02d}_view_{view_idx:02d}.png")
                cv2.imwrite(individual_path, individual_image)
        
        # Save scene image
        scene_path = f"{output_dir}/view_{view_idx:03d}.png"
        cv2.imwrite(scene_path, scene_image)
        
        # Record image path and camera position for MCP
        image_paths.append(scene_path)
        camera_positions.append(camera.position)
    
    # Generate MCP context
    mcp_context = generate_mcp_context(objects, image_paths, camera_positions)
    
    # Add object relations to MCP context
    mcp_context["scene"]["relations"] = object_relations
    
    # Add individual object images to MCP context if generated
    if generate_individual:
        mcp_context["scene"]["individual_objects"] = []
        for obj_idx, obj in enumerate(objects):
            obj_images = []
            for view_idx in range(num_views):
                img_path = os.path.join("individual_objects", f"obj_{obj_idx:02d}_view_{view_idx:02d}.png")
                obj_images.append(img_path)
            
            mcp_context["scene"]["individual_objects"].append({
                "object_id": obj_idx,
                "class_name": obj.class_name,
                "images": obj_images
            })
    
    return mcp_context

if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 2D images from 3DETR output")
    parser.add_argument("--input", required=True, help="Path to 3DETR output JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--num_views", type=int, default=8, help="Number of viewpoints to render")
    parser.add_argument("--mcp_output", default=None, help="Path to save MCP context JSON")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the generated images (width and height)")
    parser.add_argument("--no_individual", action="store_true", help="Do not generate individual object images")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse 3DETR output
    objects = parse_3detr_output(args.input)
    
    # Generate images and MCP context
    mcp_context = generate_object_images(
        objects, 
        args.output_dir, 
        args.num_views,
        generate_individual=not args.no_individual,
        image_size=args.image_size
    )
    
    # Save MCP context
    mcp_output_path = args.mcp_output or os.path.join(args.output_dir, "mcp_context.json")
    save_mcp_context(mcp_context, mcp_output_path)
    
    print(f"Generated {args.num_views} scene views in {args.output_dir}")
    if not args.no_individual:
        print(f"Generated {len(objects) * args.num_views} individual object images")
    print(f"Saved MCP context to {mcp_output_path}")
    print(f"Found {len(objects)} objects in the scene")
