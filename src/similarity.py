# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

from transformers import AutoTokenizer, AutoModel
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoImageProcessor, ResNetModel

import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Text_Embedding():
    def __init__(self):
        """
        Incarcam modelele si tokenizer-ul

        Pentru performanta mai buna, le pastram in memorie intre apeluri

        """
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def embed_phrases(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        np_sentence_embeddings = sentence_embeddings.to("cpu").to("cpu").detach().numpy()
        return np_sentence_embeddings


class Image_Embedding:
    def __init__(self):
        # self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        # self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")

    def embed_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        np_hidden_state = torch.flatten(last_hidden_states, 1).to("cpu").to("cpu").detach().numpy()
        return np_hidden_state

