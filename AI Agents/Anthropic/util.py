import os
import ollama

def llm_call(prompt:str, model="llama3.2:1b" ):
    """
    """
    messages = [ {"role": "user",
                "content" : prompt} ]
    
    response = ollama.chat(model=model,messages=messages,stream= False, options= {'num_predict':512, 'temperature':0.1})
    
    return response['message']['content'] 

def extract_xml(text:str, tag:str):

    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    
    return match.group(1) if match else ""