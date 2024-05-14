
import streamlit as st


import requests

API_URL = "https://api-inference.huggingface.co/models/shivanikerai/TinyLlama-1.1B-Chat-v1.0-sku-title-ner-generation-reversed-v1.0"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# Function to perform NER on the title
def ner_for_title(title):
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_in, E_in = "[Title]", "[/Title]"
    # Format your prompt template
    prompt = f"""{B_INST} {B_SYS} You are a helpful assistant that provides accurate and concise responses. {E_SYS}\nExtract named entities from the given product title. Provide the output in JSON format.\n{B_in} {title.strip()} {E_in}\n{E_INST}\n\n### NER Response:\n{{"{title.split()[0].lower()}"""
    output = query({
    "inputs": prompt,
    "parameters": {"return_full_text":False,},
    "options":{"wait_for_model": True}
    })

    return eval('{"'+title.split()[0]+output[0]['generated_text'])



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
	st.write("created by NB Nitin Bhayana")
        
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
        st.subheader("General Parameters")
        st.write("Length of Title           : ", len(title_input))
        st.write("Count of words            : ", len(title_input.split()))
        st.write("Count of attributes       : ", len(ner_result))
        st.write("Count of alpha-numeric    : ",sum(char.isalnum() for char in title_input))
        st.write("Count of non alpha-numeric: ",len(title_input)-sum(char.isalnum() or char == ' ' for char in title_input))
if __name__ == "__main__":
    main()

