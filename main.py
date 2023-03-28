import openai
import os
import json

from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, download_loader, LLMPredictor, PromptHelper

os.environ['OPENAI_API_KEY'] = "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"
documents = SimpleDirectoryReader('./data').load_data()
llm_predictor = LLMPredictor(llm=OpenAI(temperature=2.0, model_name="text-davinci-002"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

custom_LLM_index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = []

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history[-5:]])
        prompt += f"\nUser: {user_input}"
        response = index.query(user_input)

        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)

# Wikipedia Loader Index
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
wikidocs = loader.load_data(pages=['Formula One'])

# Beautiful Soup Website Loader Index
BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()
soupdocs = loader.load_data(urls=['https://www.formula1.com/'])

# Defining Index Type
index = GPTSimpleVectorIndex(wikidocs)
bot = Chatbot("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", index=index)

while True:
    user_input = input("")
    response = bot.generate_response(user_input)
    print(f"{response['content']}")