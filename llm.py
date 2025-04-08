from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import re
import ast
# Load environment variables from .env file
load_dotenv()


api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=api_key)
prompt = """You are an event extraction expert. Given a text, extract the event triggers. You should return a list of events in the last line with format:
The events are: [...]. 
Each event should be a dictionary with the following keys: 'event_type', 'trigger_word', 'event_time', 'event_location', 'event_participants'. 
The values for these keys should be extracted from the text. If any of the keys are not present in the text, return None for that key.
For example:
1. If the text is "John and Mary met at the park on Monday", the output should be:
The events are: [{{'event_type': 'meeting', 'trigger_word': 'met', 'event_time': 'Monday', 'event_location': 'park', 'event_participants': ['John', 'Mary']}}]
2. If the text is "The July 2006 earthquake was also centered in the Indian Ocean, from the coast of Java, and had a duration of more than three minutes.", the output should be:
The events are: [{{'event_type': 'catastrophe', 'trigger_word': 'earthquake', 'event_time': 'July 2006', 'event_location': 'Indian Ocean', 'event_participants': None}}, 
                {{'event_type': 'placing', 'trigger_word': 'centered', 'event_time': 'July 2006', 'event_location': 'Indian Ocean', 'event_participants': None}}]
3. If the text does not contain any events, return an empty list.
The events are: []

Now, please extract the events from the following text:
{content}
"""

def response_to_string(response, idx=0):
    if idx > len(response.candidates):
        idx = 0
    output = []
    
    for part in response.candidates[idx].content.parts:
        if part.text is not None:
            output.append(part.text)
        if part.executable_code is not None:
            output.append(f"```python\n{part.executable_code.code}\n```")  # Định dạng mã code
        if part.code_execution_result is not None:
            output.append(f"Output:\n{part.code_execution_result.output}")
        if part.inline_data is not None:
            output.append("[Hình ảnh được nhúng]")  # Không thể hiển thị trực tiếp hình ảnh trong chuỗi

    return "\n".join(output)

def extract_response(text:str):
    match = re.search(r'The events are:\s*(\[.*\])', text, re.DOTALL)

    if match:
        events_str = match.group(1)
        try:
            events_list = ast.literal_eval(events_str)
            return events_list
        
        except ValueError as e:
            print(f"Error parsing events: {e}")
            return None
    else:
        print("No events found in the response.")
        return None

def extract_event_gemini(text:str, model="gemini-2.0-flash", candidate=1):
    """
    Extract events from text using Google Gemini API.
    """
    # Gen answer
    response = client.models.generate_content(
      model=model,
      contents=prompt.format(content=text),
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
        candidate_count=candidate
      )
    )

    res = []
    for idx in range(len(response.candidates)): 
        response_string = response_to_string(response, idx)
        event_list = extract_response(response_string)
        res.append(event_list)

    # Filter out None values and return the first non-None result
    res = [item for item in res if item is not None]
    
    return res
    