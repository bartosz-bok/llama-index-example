from huggingface_hub import hf_hub_download

from common import MODELS_DIR

# The GGUF file from the HuggingFace repository
REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
LLM_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if __name__ == '__main__':

    print("Downloading model...")

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=LLM_FILENAME,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"Model saved as: {model_path}")
