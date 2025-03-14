from huggingface_hub import snapshot_download
from common import MODELS_DIR

# Embedding model repository from HuggingFace
REPO_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_FILENAME = "bge-small-en-v1.5"

if __name__ == '__main__':

    print("Downloading embedding model...")

    model_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODELS_DIR / EMBEDDING_MODEL_FILENAME,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors", "*.onnx"],
    )

    print(f"Embedding model saved at: {model_path}")
