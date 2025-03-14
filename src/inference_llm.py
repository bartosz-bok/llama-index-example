from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

from common import MODELS_DIR
from src.download_embedding_model import EMBEDDING_MODEL_FILENAME
from src.download_llm import LLM_FILENAME

if __name__ == "__main__":
    # Define model paths
    llm_path = str(MODELS_DIR / LLM_FILENAME)
    embedding_model_path = str(MODELS_DIR / EMBEDDING_MODEL_FILENAME)

    # Initialize local LLM
    Settings.llm = LlamaCPP(model_path=llm_path, temperature=0.7)

    # Set the local embeddings model
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)

    # Documents to be indexed
    documents = [
        Document(text=
                 "Michael Jeffrey Jordan (ur. 17 lutego 1963 w Nowym Jorku) – amerykański koszykarz występujący na "
                 "pozycji rzucającego obrońcy, sześciokrotny mistrz NBA, dwukrotny złoty medalista olimpijski, członek "
                 "Koszykarskiej Galerii Sław.")
    ]

    # Set index
    index = VectorStoreIndex.from_documents(documents)

    # Set query engine
    query_engine = index.as_query_engine()

    # Query to index
    query = "O czym jest dokument?"
    response = query_engine.query(query)

    print("Model reposnse:")
    print(response)
