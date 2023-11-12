import similarity
import cv2

def test_basic_text_embedding():
    simobj = similarity.Text_Embedding()
    sentences = ['This is an example sentence', 'Each sentence is converted']
    sentence_embeddings = simobj.embed_phrases(sentences)
    assert sentence_embeddings.shape[0] == len(sentences)
    print(f"Embedding dimensions: {sentence_embeddings.shape}")



def test_basic_image_embedding():
    simobj = similarity.Image_Embedding()
    img = cv2.imread("images/FudanPed00059.png")
    embed = simobj.embed_image(img)
    print(embed.shape)