


def mount_drive():
    """
    Mounts drive for colab
    """
    from google.colab import drive
    drive.mount('/content/drive')
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



    