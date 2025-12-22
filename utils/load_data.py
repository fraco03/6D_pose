


def mount_drive(path='/content/drive'):
    """
    Mounts drive for colab
    """
    from google.colab import drive
    drive.mount(path)
    print("✅ Drive mounted at /content/drive")


def download_dataset():
    """
    Downloads LineMod preprocessed dataset
    Works on both scripts and Jupyter notebooks
    """
    global DATASET_PATH
    import gdown
    import zipfile
    import os
    
    url = "https://drive.google.com/file/d/1YFUra533pxS_IHsb9tB87lLoxbcHYXt8/view?usp=drive_link"
    output = "Linemod_preprocessed.zip"

    if os.path.exists('./Linemod_preprocessed'):
        print(f"✅ Dataset already exists at ./Linemod_preprocessed")
        DATASET_PATH = './Linemod_preprocessed'
        return
    
    print("📥 Downloading dataset from Google Drive...")
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
        print("✅ Download complete")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return
    
    print("📦 Extracting dataset...")
    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete")
        
        # Clean up zip file
        os.remove(output)
        print("🗑️  Cleaned up zip file")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return
    
    DATASET_PATH = './Linemod_preprocessed'
    print(f"\n✅ Dataset ready at: {DATASET_PATH}")


def root_dataset_path():
    """
    Returns the root dataset path
    """
    return DATASET_PATH


def load_model_data(checkpoint_location: str, map_location: str, model_key: str = 'model_state_dict', allow_multiple_devices=False) -> dict:
    """
    Loads model points from a .pth file.

    Args:
        checkpoint_location (str): Path to the .pth file containing model weights.

    Returns:
        dict: A dictionary containing checkpoint data
    """
    import torch
    from collections import OrderedDict

    model_data = torch.load(checkpoint_location, map_location=map_location)

    state_dict = model_data.get(model_key, model_data)

    if not allow_multiple_devices:
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix if present
            new_state_dict[new_key] = value

        model_data[model_key] = new_state_dict
    return model_data

def load_model(checkpoint_location: str, device: str, model_class, model_key: str = 'model_state_dict', **kwargs):
    model_data = load_model_data(checkpoint_location=checkpoint_location, map_location=device, model_key=model_key)

    model = model_class(**kwargs)
    model.load_state_dict(model_data[model_key])
    model.to(device)
    return model


    


    
