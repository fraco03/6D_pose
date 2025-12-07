# test_dataset.py
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'pose_rgb'))

from dataset import LineModPoseDataset
from pose_utils import quaternion_to_rotation_matrix

def test_dataset_basic():
    """Test basic dataset functionality"""
    print("=" * 60)
    print("🧪 TEST 1: Basic Dataset Loading")
    print("=" * 60)
    
    # Test with small dataset and 2 objects
    dataset = LineModPoseDataset(
        root_dir='Linemod_preprocessed_small',
        split='train',
        object_ids=[1, 2],  # Ape and Benchvise
        image_size=(224, 224),
        normalize=True
    )
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    return dataset


def test_single_sample(dataset):
    """Test loading a single sample"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Single Sample Loading")
    print("=" * 60)
    
    sample = dataset[0]
    
    print(f"\n📦 Sample 0 contents:")
    print(f"   image shape:        {sample['image'].shape}")
    print(f"   rotation:      {sample['rotation'].shape} -> {sample['rotation'].numpy()}")
    print(f"   translation:        {sample['translation'].shape} -> {sample['translation'].numpy()}")
    print(f"   object_id:          {sample['object_id']}")
    print(f"   class_idx:          {sample['class_idx']}")
    print(f"   cam_K shape:        {sample['cam_K'].shape}")
    print(f"   img_id:             {sample['img_id']}")
    
    # Verify quaternion is normalized
    quat_norm = torch.norm(sample['rotation']).item()
    print(f"\n✅ Quaternion norm: {quat_norm:.6f} (should be ~1.0)")
    
    assert abs(quat_norm - 1.0) < 1e-4, f"Quaternion not normalized! Norm = {quat_norm}"
    
    return sample


def test_quaternion_conversion(sample):
    """Test quaternion to rotation matrix conversion"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Quaternion Conversion")
    print("=" * 60)
    
    quat = sample['rotation'].numpy()
    R_reconstructed = quaternion_to_rotation_matrix(quat)
    
    print(f"\n📐 Reconstructed Rotation Matrix:")
    print(R_reconstructed)
    
    # Check if it's a valid rotation matrix
    det = np.linalg.det(R_reconstructed)
    orthogonality = np.abs(R_reconstructed @ R_reconstructed.T - np.eye(3)).max()
    
    print(f"\n✅ Rotation matrix properties:")
    print(f"   Determinant: {det:.6f} (should be ~1.0)")
    print(f"   Orthogonality error: {orthogonality:.6e} (should be ~0)")
    
    assert abs(det - 1.0) < 1e-4, f"Invalid rotation matrix! det = {det}"
    assert orthogonality < 1e-4, f"Not orthogonal! error = {orthogonality}"


def test_dataloader():
    """Test DataLoader with batching"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: DataLoader Batching")
    print("=" * 60)
    
    dataset = LineModPoseDataset(
        root_dir='Linemod_preprocessed_small',
        split='test',
        object_ids=[1, 2],
        image_size=(224, 224),
        normalize=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    
    print(f"\n📦 Batch contents:")
    print(f"   images:        {batch['image'].shape}")
    print(f"   rotation: {batch['rotation'].shape}")
    print(f"   translation:   {batch['translation'].shape}")
    print(f"   object_ids:    {batch['object_id']}")
    print(f"   cam_K:         {batch['cam_K'].shape}")
    
    print(f"\n✅ DataLoader works correctly!")
    
    return dataloader


def test_all_objects():
    """Test loading all objects"""
    print("\n" + "=" * 60)
    print("🧪 TEST 5: All Objects")
    print("=" * 60)
    
    dataset = LineModPoseDataset(
        root_dir='Linemod_preprocessed_small',
        split='test',
        object_ids=None,  # All objects
        image_size=(224, 224),
        normalize=True
    )
    
    # Count samples per object
    object_counts = {}
    for sample in dataset.samples:
        obj_id = sample['object_id']
        object_counts[obj_id] = object_counts.get(obj_id, 0) + 1
    
    print(f"\n📊 Samples per object:")
    for obj_id, count in sorted(object_counts.items()):
        class_name = dataset.get_class_name(obj_id)
        print(f"   Object {obj_id:02d} ({class_name:12s}): {count:4d} samples")
    
    print(f"\n✅ Total: {len(dataset)} samples across {len(object_counts)} objects")


def visualize_sample(dataset, idx=0):
    """Visualize a sample"""
    print("\n" + "=" * 60)
    print("🧪 TEST 6: Visualization")
    print("=" * 60)
    
    sample = dataset[idx]
    
    # Denormalize image for visualization
    img = sample['image'].numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    
    if dataset.normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
    
    img = np.clip(img, 0, 1)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    
    # Add info
    obj_id = sample['object_id']
    class_name = dataset.get_class_name(dataset.id_to_class[obj_id])
    plt.title(f"Object {obj_id}: {class_name}\nImage ID: {sample['img_id']}", 
              fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig('test_sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to 'test_sample_visualization.png'")
    plt.close()


def test_model_integration():
    """Test model with dataset"""
    print("\n" + "=" * 60)
    print("🧪 TEST 7: Model Integration with cam_K")
    print("=" * 60)
    
    try:
        from model import ResNetQuaternion
        
        # Create model
        model = ResNetQuaternion(freeze_backbone=True)
        model.eval()
        
        print("✅ Model created successfully!")
        
        # Create dataset and dataloader
        dataset = LineModPoseDataset(
            root_dir='Linemod_preprocessed_small',
            split='train',
            object_ids=[1],
            image_size=(224, 224),
            normalize=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        images = batch['image']
        cam_K = batch['cam_K']
        gt_rot = batch['rotation']
        gt_trans = batch['translation']
        
        print(f"\n📦 Batch shapes:")
        print(f"   Images:      {images.shape}")
        print(f"   cam_K:       {cam_K.shape}")
        print(f"   GT rotation: {gt_rot.shape}")
        print(f"   GT trans:    {gt_trans.shape}")
        
        # Forward pass
        with torch.no_grad():
            pred_rot, pred_trans = model(images, cam_K)
        
        print(f"\n🔮 Model predictions:")
        print(f"   Pred rotation: {pred_rot.shape}")
        print(f"   Pred trans:    {pred_trans.shape}")
        print(f"   Quaternion norms: {torch.norm(pred_rot, dim=1)}")
        
        # Verify quaternion normalization
        quat_norms = torch.norm(pred_rot, dim=1)
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), \
            "Quaternions not normalized!"
        
        print(f"\n✅ Model forward pass successful!")
        print(f"✅ All quaternions properly normalized!")
        
    except ImportError:
        print("⚠️  Warning: Could not import model. Skipping model test.")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        raise


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("STARTING DATASET TESTS")
    print("🚀" * 30)
    
    try:
        # Test 1: Basic loading
        dataset = test_dataset_basic()
        
        # Test 2: Single sample
        sample = test_single_sample(dataset)
        
        # Test 3: Quaternion conversion
        test_quaternion_conversion(sample)
        
        # Test 4: DataLoader
        test_dataloader()
        
        # Test 5: All objects
        test_all_objects()
        
        # Test 6: Visualization
        visualize_sample(dataset, idx=0)
        
        # Test 7: Model integration
        test_model_integration()
        
        print("\n" + "✅" * 30)
        print("ALL TESTS PASSED!")
        print("✅" * 30 + "\n")
        
    except Exception as e:
        print("\n" + "❌" * 30)
        print(f"TEST FAILED: {e}")
        print("❌" * 30 + "\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
    # test_all_objects()