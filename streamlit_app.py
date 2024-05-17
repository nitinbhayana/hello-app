import os
import streamlit as st
import numpy as np
import math
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter



import requests

API_URL_ner = "https://api-inference.huggingface.co/models/shivanikerai/TinyLlama-1.1B-Chat-v1.0-sku-title-ner-generation-reversed-v1.0"
API_URL_suggest = "https://api-inference.huggingface.co/models/nitinbhayana/TinyLlama-1.1B-Chat-v1.0-title-suggestion-v1.0"
headers = {"Authorization": "Bearer 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"}

# nltk_data_dir = "./resources/nltk_data_dir/"
# if not os.path.exists(nltk_data_dir):
#     os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.clear()
# nltk.data.path.append(nltk_data_dir)
# nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')




def query(API_URL,payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def len_title(title):
    x=len(title)
    mean,sigma=190.5, 80
    gaussian_value = np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))
    if x < 50 or x > 250:
        return 0
    return round(gaussian_value, 2)

def count_words_by_length(words, min_len, max_len=None):
    if max_len is None:
        score=sum(1 for word in words if len(word) >= min_len)
    else:
        score= sum(1 for word in words if min_len <= len(word) <= max_len)
    return round(score, 2)

def simplicity_score(title):
    words=title.split()
    a = count_words_by_length(words, 1, 3)
    b = count_words_by_length(words, 4, 6)
    c = count_words_by_length(words, 7, 10)
    d = count_words_by_length(words, 11)
    score= (1-(3*a+b+2*c+4*d)/(10*(a+b+c+d))) #* (1 / (1 + math.log((a+b+c+d)+ 1)))+0.45
    return round(score, 2)

# def duplicacy(title):
#     # Tokenize the title into words
#     words = word_tokenize(title.lower())
    
#     # POS tagging
#     pos_tags = pos_tag(words)
    
#     # Filter out conjunctions (CC tag in Penn Treebank tag set)
#     non_conjunction_words = [word for word, tag in pos_tags if tag != 'CC' or tag != 'IN']
    
#     # Count the occurrences of each non-conjunction word
#     word_counts = Counter(non_conjunction_words)
    
#     # Count the duplicates
#     duplicate_non_conjunction_count = sum(1 for count in word_counts.values() if count > 1)
    
#     total_words=len(non_conjunction_words)
    
#     if total_words == 0:
#         return 1  # Maximum score if there are no words
#     score = 1 - (duplicate_non_conjunction_count / total_words)
    
#     return round(score, 2)

def emphasis_score(title):
    words = title.split()
    uppercase_count = sum(1 for word in words if word[0].isupper())
    x=uppercase_count/len(words)*100
    if uppercase_count > 0:
        if x <= 60:
            mean, sigma = 60, 30
        else:
            mean, sigma = 60, 10
        score = np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))
    
    else:
        score=0
    return round(score, 2)

# Function to perform NER on the title
def ner_for_title(title):
    
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_in, E_in = "[Title]", "[/Title]"
    # Format your prompt template
    prompt = f"""{B_INST} {B_SYS} You are a helpful assistant that provides accurate and concise responses. {E_SYS}\nExtract named entities from the given product title. Provide the output in JSON format.\n{B_in} {title.strip()} {E_in}\n{E_INST}\n\n### NER Response:\n{{"{title.split()[0].lower()}"""
    output = query(API_URL_ner,{
    "inputs": prompt,
    "parameters": {"return_full_text":False,},
    "options":{"wait_for_model": True}
    })

    return eval('{"'+title.split()[0]+output[0]['generated_text'])

def suggest_title(title):
    prompt=f"""<s>[INST] <<SYS>> You are a helpful assistant that provides accurate and concise responses. <</SYS>>
Create a new, easy to read, and error free title for a given Ecommerce product title.
[Title] {title} [/Title]
[/INST]
### Suggested Title:"""

    output = query(API_URL_suggest,{
    "inputs": prompt,
    "parameters": {"return_full_text":False,},
    "options":{"wait_for_model": True}
    })

    return output[0]['generated_text']

# Streamlit app layout
def main():
    
    if ner_for_title("HP laptop"):
      st.title("Product Attributes Decoding")
    
    # Input text box for the product title
      title_input = st.text_input("Enter product title:")
    
    if st.button("Submit"):
        # Perform NER on the input title
        ner_result = ner_for_title(title_input)
        #st.write(ner_result)
        # Display the title with NER annotations
        st.subheader("Artificial Intelligence")
        st.write("created by NB")
        
        # Start from the original title and replace phrases one by one
        annotated_title = title_input
        
        # Sort entities by their start position to handle them in the correct sequence
        entities = sorted(ner_result, key=lambda x: title_input.lower().find(x.lower()))
        
        # Apply HTML tags to each entity found in the title
        for entity in entities:
            start_index = title_input.lower().find(entity.lower())
            if start_index != -1:  # Only proceed if the entity is found
                original_text = title_input[start_index:start_index + len(entity)]
                # Replace the original text with the annotated version in the title
                annotated_title = annotated_title.replace(original_text, 
                    f"{original_text}<span style='color:green;'>({ner_result[entity]})</span>", 1)
        
        # Display the fully annotated title using HTML to allow styling
        st.markdown(annotated_title, unsafe_allow_html=True)

        st.subheader("Suggested Title")
        suggest_result = suggest_title(title_input)
        st.write(suggest_result)

        
        st.subheader("General Parameters")
        # st.write("Length of Title           : ", len(title_input))
        # st.write("Count of words            : ", len(title_input.split()))
        # st.write("Count of attributes       : ", len(ner_result))
        # st.write("Count of alpha-numeric    : ",sum(char.isalnum() for char in title_input))
        # st.write("Count of non alpha-numeric: ",len(title_input)-sum(char.isalnum() or char == ' ' for char in title_input))
        
        st.write("Title length score",len_title(title_input))
        st.write("Title simplicity", simplicity_score(title_input))
        st.write("Title emphasis",  emphasis_score(title_input)) #duplicacy(title),
if __name__ == "__main__":
    main()
