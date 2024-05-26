from openai import OpenAI
from llamaapi import LlamaAPI
from dotenv import load_dotenv
import os
import re

def get_predictionLLama_Mistral(llama, content, labels, model):
    try:
        # Prepare API request JSON
        api_request_json = {
            "model": model,
            "messages": [
                {"role": "user", "content": content},
            ]
        }
        response = llama.run(api_request_json)
        response_data = response.json()
        predicted_label = None
        for choice in response_data['choices']:
            if choice['message']['role'] == 'assistant':
                predicted_label = choice['message']['content']
                break

        found_label = None
        for label in labels:
            if label in predicted_label:
                found_label = label
                break
            else:
                found_label = 'unknown' # Return "unknown" if no label is found

        return found_label
    except Exception as e:
        raise Exception("Error processing prediction: {}".format(str(e)))



def get_predictionGPT(content, labels, model):
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}]
        )
        prediction = completion.choices[0].message.content.lower()

        for label in labels:
            pattern = re.escape(label)
            if re.search(pattern, prediction):
                return label
        return "unknown"  # Return "unknown" if no label is found
    except Exception as e:
        raise Exception("Error processing prediction: {}".format(str(e)))


def get_prediction(text_data, selected_model):
    labels = ['data profiling', 'classification', 'correlation', 'anomaly detection', 'clustering', 'causal inference', 'association rules', 'regression', 'forecasting']
    content = f"""Classes: {labels}\nText: {text_data}\n\nClassify the text into one of the above classes."""

    load_dotenv() 
    if "llama" in selected_model or "mixtral" in selected_model or "mistral" in selected_model:
        llama_key = os.getenv("LlamaAPI_KEY")
        llama = LlamaAPI(llama_key)
        prediction = get_predictionLLama_Mistral (llama, content, labels, model=selected_model)
    elif "gpt" in selected_model:
        prediction = get_predictionGPT (content, labels, model=selected_model)
    
    return prediction


