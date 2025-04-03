#!/usr/bin/env python
# Convert OBJ file to ScanNet format for 3DETR

import os
import sys
import argparse
import numpy as np
import trimesh
import json
from pathlib import Path
import torch

def load_obj_as_pointcloud(obj_file, num_points=40000):
    """
    OBJ 파일을 포인트 클라우드로 로드
    
    Args:
        obj_file: OBJ 파일 경로
        num_points: 샘플링할 포인트 수
    
    Returns:
        points: (N, 3) 배열의 포인트 클라우드
    """
    print(f"Loading OBJ file: {obj_file}")
    mesh = trimesh.load(obj_file)
    
    # 메시가 너무 큰 경우 크기 조정
    mesh_extent = mesh.extents.max()
    if mesh_extent > 10:
        scale_factor = 5.0 / mesh_extent
        mesh.apply_scale(scale_factor)
        print(f"Scaled mesh by factor: {scale_factor}")
    
    # 메시에서 포인트 클라우드 샘플링
    points = mesh.sample(num_points)
    
    # 중심점을 (0,0,0)으로 이동
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    points = points - center
    
    print(f"Sampled {points.shape[0]} points from mesh")
    return points

def generate_simple_bboxes(points, num_clusters=5):
    """
    간단한 클러스터링을 기반으로 바운딩 박스 생성
    
    Args:
        points: 포인트 클라우드
        num_clusters: 클러스터(객체) 수
    
    Returns:
        bboxes: 바운딩 박스 배열 [cx, cy, cz, dx, dy, dz, class_id]
    """
    from sklearn.cluster import KMeans
    
    print(f"Generating {num_clusters} bounding boxes using K-means clustering")
    
    # 포인트 수가 너무 많으면 서브샘플링하여 클러스터링 가속화
    subsample_size = min(10000, points.shape[0])
    indices = np.random.choice(points.shape[0], subsample_size, replace=False)
    subsampled_points = points[indices]
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(subsampled_points)
    
    # 모든 포인트에 클러스터 레이블 할당
    labels = kmeans.predict(points)
    
    bboxes = []
    for i in range(num_clusters):
        cluster_points = points[labels == i]
        
        # 클러스터에 포인트가 너무 적으면 건너뛰기
        if cluster_points.shape[0] < 100:
            continue
            
        # 바운딩 박스 계산
        min_point = np.min(cluster_points, axis=0)
        max_point = np.max(cluster_points, axis=0)
        
        center = (min_point + max_point) / 2
        size = max_point - min_point
        
        # 크기가 너무 작은 박스는 건너뛰기
        if np.any(size < 0.1):
            continue
            
        # 랜덤 클래스 ID 할당 (0-17 사이, ScanNet 클래스)
        class_id = np.random.randint(0, 18)
        
        bbox = np.concatenate([center, size, [class_id]])
        bboxes.append(bbox)
    
    bboxes = np.array(bboxes)
    print(f"Generated {len(bboxes)} valid bounding boxes")
    return bboxes

def create_scannet_scene_from_obj(obj_file, output_dir, scene_id="scene0000_00", num_points=40000, num_clusters=5):
    """
    OBJ 파일에서 ScanNet 형식의 씬 데이터 생성
    
    Args:
        obj_file: OBJ 파일 경로
        output_dir: 출력 디렉토리
        scene_id: 씬 ID
        num_points: 샘플링할 포인트 수
        num_clusters: 생성할 객체 클러스터 수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # OBJ 파일에서 포인트 클라우드 로드
    points = load_obj_as_pointcloud(obj_file, num_points)
    
    # 바운딩 박스 생성
    bboxes = generate_simple_bboxes(points, num_clusters)
    
    # 포인트에 가상 색상 추가 (회색 기본값)
    colors = np.ones_like(points) * 128  # 회색 [128, 128, 128]
    vertices = np.concatenate([points, colors], axis=1)
    
    # 가상 인스턴스 및 시맨틱 레이블 생성
    instance_labels = np.zeros(points.shape[0], dtype=np.int32)
    semantic_labels = np.zeros(points.shape[0], dtype=np.int32)
    
    # 가장 가까운 바운딩 박스 중심에 따라 포인트에 레이블 할당
    for i, bbox in enumerate(bboxes):
        center = bbox[:3]
        # 각 포인트와 박스 중심 사이의 거리 계산
        distances = np.linalg.norm(points - center, axis=1)
        # 확률적으로 일부 포인트를 이 인스턴스에 할당
        mask = (distances < np.random.uniform(0.5, 1.5)) & (np.random.random(points.shape[0]) < 0.7)
        instance_labels[mask] = i + 1  # 인스턴스 ID는 1부터 시작
        semantic_labels[mask] = int(bbox[6])  # 시맨틱 ID = 클래스 ID
    
    # 데이터 저장
    np.save(os.path.join(output_dir, f"{scene_id}_vert.npy"), vertices)
    np.save(os.path.join(output_dir, f"{scene_id}_ins_label.npy"), instance_labels)
    np.save(os.path.join(output_dir, f"{scene_id}_sem_label.npy"), semantic_labels)
    np.save(os.path.join(output_dir, f"{scene_id}_bbox.npy"), bboxes)
    
    # 학습/테스트 분할 파일 생성
    with open(os.path.join(output_dir, "scannetv2_train.txt"), "w") as f:
        f.write(f"{scene_id}\n")
        
    with open(os.path.join(output_dir, "scannetv2_val.txt"), "w") as f:
        f.write(f"{scene_id}\n")
    
    print(f"Created ScanNet format scene in {output_dir}")
    print(f"Use the following command to test 3DETR on this data:")
    print(f"python main.py --test_only --test_ckpt /path/to/checkpoint.pth --dataset_name scannet --dataset_root_dir {output_dir}")
    
    return {
        "scene_id": scene_id,
        "num_points": points.shape[0],
        "num_bboxes": bboxes.shape[0],
        "output_dir": output_dir
    }

def main():
    parser = argparse.ArgumentParser(description="Convert OBJ file to ScanNet format for 3DETR")
    parser.add_argument("--obj_file", required=True, help="Path to OBJ file")
    parser.add_argument("--output_dir", required=True, help="Output directory for ScanNet format data")
    parser.add_argument("--scene_id", default="scene0000_00", help="Scene ID")
    parser.add_argument("--num_points", type=int, default=40000, help="Number of points to sample")
    parser.add_argument("--num_objects", type=int, default=5, help="Number of objects to generate")
    
    args = parser.parse_args()
    
    create_scannet_scene_from_obj(
        args.obj_file, 
        args.output_dir,
        args.scene_id,
        args.num_points,
        args.num_objects
    )

if __name__ == "__main__":
    main() 