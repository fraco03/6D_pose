def load_repo():
    """
    Loads the repository from github, installs dependencies, and adds modules to sys.path
    Works on both scripts and Jupyter notebooks
    """
    import subprocess
    import os
    import sys
    
    repo_url = "https://github.com/fraco03/6D_pose.git"
    repo_dir = "6D_pose"
    
    print("📦 Cloning repository...")
    try:
        subprocess.run(["git", "clone", repo_url], check=True, capture_output=False)
        print("✅ Repository cloned")
    except subprocess.CalledProcessError:
        print("⚠️  Repository already exists or clone failed")
    
    print("📥 Installing dependencies...")
    # try:
    #     # Use cwd to ensure pip runs in the correct directory
    #     subprocess.run(
    #         ["pip", "install", "-r", "requirements.txt"],
    #         cwd=repo_dir,
    #         check=True,
    #         capture_output=False
    #     )
    #     print("✅ Dependencies installed")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Failed to install dependencies: {e}")
    # except FileNotFoundError:
    #     print("❌ Repository directory not found")
    
    # Add repository to sys.path for imports
    print("🔧 Adding modules to Python path...")
    repo_path = os.path.abspath(repo_dir)
    
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        print(f"✅ Added {repo_path} to sys.path")
    else:
        print("⚠️  Repository already in sys.path")
    
    # Verify key modules are accessible
    print("🔍 Verifying module imports...")
    try:
        # Test imports
        from src.pose_rgb import dataset, model, loss
        from src.detection import yolo_utils
        from utils import load_data
        print("✅ All modules accessible!")
    except ImportError as e:
        print(f"⚠️  Some modules may not be accessible: {e}")
    
    print("\n✅ Setup complete! You can now import modules:")
    # print("   from src.pose_rgb.dataset import LineModPoseDataset")
    # print("   from src.pose_rgb.model import ResNetQuaternion")
    # print("   from src.pose_rgb.loss import PoseLoss")
    # print("   from utils.load_data import download_dataset, mount_drive")