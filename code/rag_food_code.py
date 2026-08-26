import argparse
import warnings
import os
import logging
import requests
import pandas as pd
import numpy as np
import time
import base64
from operator import itemgetter

from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from config import MODELS, LOCAL_VLLM_BASE_URL, LOCAL_VLLM_EMBED_BASE_URL

try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever


def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def configure_chat_openai(model_key, models):
    """Create a chat model for text or image prompts, served locally via vLLM on GPU 0
    (see code/analyze_image.py). It's a reasoning model, so it needs a larger token budget
    to cover its <think> block plus the actual answer."""
    return ChatOpenAI(
        model=models[model_key],
        temperature=0.1 if model_key == 'llm_vision' else 0.3,
        max_tokens=1024 if model_key == 'llm_vision' else 3072,
        base_url=LOCAL_VLLM_BASE_URL,
        api_key='not-needed',
    )

def initialize_clients(models):
    llm = configure_chat_openai('llm', models)
    llm_vision = configure_chat_openai('llm_vision', models)
    embedding = OpenAIEmbeddings(
        model=models['embedding'],
        base_url=LOCAL_VLLM_EMBED_BASE_URL,
        api_key='not-needed',
        check_embedding_ctx_length=False,
    )
    return llm, llm_vision, embedding

def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def setup_vector_database(data, embedding):
    attempts = 0
    max_attempts = 5  # You can set this to the number of retry attempts you find appropriate
    while attempts < max_attempts:
        try:
            vectordb = Chroma.from_documents(documents=data, embedding=embedding, collection_name="openai_embed")
            return vectordb  # Successful creation, return the database
        except Exception as e:
            logging.error(f"Failed to setup vector database: {e}")
            attempts += 1
            time.sleep(180)  # Wait for 30 seconds before retrying
            
    # If the loop completes without returning, log final failure and optionally raise an exception
    logging.critical("Failed to initialize vector database after multiple attempts.")
    raise Exception("Failed to initialize vector database after multiple attempts.")

# Helper functions for message generation and checkpointing
def get_messages_from_url(url_str):
    """Generate a sequence of messages for a URL containing an image."""
    if os.path.isfile(url_str):
        with open(url_str, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        url_str = f"data:image/png;base64,{encoded_image}"

    return [
        SystemMessage(
            content="You are an expert at analyzing images with computer vision. "
                    "I will present you with a picture of food, which might be placed on a plate, inside a spoon, "
                    "or contained within different vessels. Your job is to accurately identify the food depicted in the image."
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": """Could you name the food shown in the image? If feasible, specify its variety
                 based on ingredients and preparation methods. Ensure your response is brief and avoid
                 speculating on uncertain details. If you can't provide assistance with this image, simply respond 
                 with 'I can't help to analyze this image.' and provide the reason on a new line."""},
                {"type": "image_url", "image_url": {"url": url_str}}
            ]
        )
    ]

def setup_retrieval_prompt():
    RETRIEVE_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given food descriptions to retrieve relevant food information from a vector 
        database. These decriptions should docus purely on the qualitative asepcts of the food, including its broader category, specific flavor profiles, ingredients, and methods of preparation. 
        By generating multiple perspectives on the food description, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Please provide these alternate food descriptions, adding the original description as the sixth entry without mentioning that it is the original description. 
        Ensure each of the six descriptions is separated by a newline.
        Original food description: {question}""",
    )
    return RETRIEVE_PROMPT

def configure_retrievers(llm, vectordb, prompt):
    # llm_chain_retriever = LLMChain(llm=llm, prompt=prompt)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), 
        llm=llm, 
        prompt=prompt
    )
    return retriever_from_llm

def setup_code_prompt_chain(retriever_from_llm, llm):
    template = """Could you identify and provide the eight-digit food code corresponding to the given food image: {url_str}, 
    considering only the context provided: {context}?

    Please provide only the eight-digit codes without any extra information. If an exact match for the food description is not available, please identify the closest equivalent. 
    Should there be no relevant food codes based on the context provided, simply reply with 'No appropriate food codes found from the context information.
    """
    CODE_PROMPT = ChatPromptTemplate.from_template(template)

    food_code_chain = (
        {
            "context": itemgetter("question") | retriever_from_llm,
            "url_str": itemgetter("url_str"),
        }
        | CODE_PROMPT
        | llm
        | StrOutputParser()
    )
    return food_code_chain

def is_integer(s):
    """Check if a string can be converted to an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False

def process_image_url(index, url, llm_vision, df, food_code_chain, results_path):
    """Process a single image URL and update the DataFrame with food descriptions and codes."""
    logging.info(f"Processing URL at index {index}: {url}")
    df['GPTFoodDescription'] = df['GPTFoodDescription'].astype(object)
    df['GPTFoodCode'] = df['GPTFoodCode'].astype(object)
    try:
        if not os.path.isfile(url):
            req_url = requests.head(url, timeout=5)
            if req_url.status_code != 200:
                raise Exception("Image URL is not accessible")
        
        IMAGE_PROMPT = get_messages_from_url(url)
        attempts = 0  # Counter for retry attempts
        max_attempts = 5 # Set the maximum number of retry attempts
        
        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                food_description = image_response.content
                food_description_splits = food_description.split('\n')
                if food_description_splits[0] == "I can't help to analyze this image.":
                    raise Exception(food_description_splits[-1])
                break
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():  # Check if the exception is due to rate limiting
                    if attempts < max_attempts - 1:
                        sleep_time = min(2 ** attempts * 30)  # Exponential backoff with a maximum wait time
                        logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds.")
                        attempts += 1
                        time.sleep(sleep_time)
                        continue  # Continue to retry after sleeping
                    else:
                        logging.error("Maximum retry attempts reached. Failing with error.")
                        raise  # Exception to indicate max attempts have been reached
                else:
                    logging.error(f"Failed to invoke llm_vision at index {index}: {e}")
                    raise  # Re-raise the exception if it's not related to rate limits
                    
        food_code_response = food_code_chain.invoke({"question": food_description, "url_str": url})
        if food_code_response == "No appropriate food codes found from the context information.":
            raise Exception(food_code_response)

        df.loc[index, 'GPTFoodDescription'] = food_description
        df.loc[index, 'GPTFoodCode'] = str(food_code_response.split('\n'))
    except Exception as e:
        df.loc[index, 'GPTFoodDescription'] = str(e)
        df.loc[index, 'GPTFoodCode'] = np.nan
        # logging.error(f"Failed to process URL at index {index}: {e}")
    finally:
        df.to_csv(results_path, index=False)
        logging.info(f"Data saved to {results_path}")

def load_checkpoint(checkpoint_path):
    """Load the last processed indices from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            content = file.read()
            if content:
                num, index = content.split(',')
                return int(num), int(index)
    return 0, 0

def save_checkpoint(num, index, checkpoint_path):
    """Save the current indices to a checkpoint file."""
    with open(checkpoint_path, 'w') as file:
        file.write(f"{num},{index}")
    logging.info(f"Checkpoint saved to {checkpoint_path} at num {num}, index {index}")

def process_image_urls(results_path, checkpoint_path, llm_vision, food_code_chain, num_iterations):
    """Process image URLs with checkpointing that includes iterations and index."""
    df_url = pd.read_csv(results_path)
    last_num, last_index = load_checkpoint(checkpoint_path)

    for num in range(last_num, num_iterations):
        start_index = last_index if num == last_num else 0
        for i in range(start_index, len(df_url)):
            url = df_url.loc[i, 'Link']
            food_code_gpt = df_url.loc[i, 'GPTFoodCode']
            if not pd.isna(food_code_gpt):
                food_code_gpt_first = food_code_gpt[2:10]
                if is_integer(food_code_gpt_first):
                    continue
                
            process_image_url(i, url, llm_vision, df_url, food_code_chain, results_path)
            save_checkpoint(num, i + 1, checkpoint_path)  # Update checkpoint after each URL
            
        last_index = 0  # Reset last_index after completing each num iteration

def main(args):
    load_dotenv()
    setup_logging(args.log_path)
    logging.info("Starting image processing script")
    llm, llm_vision, embedding = initialize_clients(MODELS)
    data = load_data(args.csv_file)
    vectordb = setup_vector_database(data, embedding)  # Ensure 'embedding' is initialized

    retrieval_prompt = setup_retrieval_prompt()
    retriever_from_llm = configure_retrievers(llm, vectordb, retrieval_prompt)
    food_code_chain = setup_code_prompt_chain(retriever_from_llm, llm)
    # Process image URLs (assuming function `process_image_urls` is properly defined)
    process_image_urls(
        args.results_file,
        args.checkpoint_file,
        llm_vision,
        food_code_chain,
        args.num_iterations,
    )
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(description="Process image URLs and update with food descriptions.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing data.")
    parser.add_argument("--checkpoint_file", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--results_file", required=True, help="Path to the results file containing image URLs.")
    parser.add_argument("--log_path", default="process_images.log", help="Path to the log file.")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of passes over the result rows.")
    
    args = parser.parse_args()
    main(args)
