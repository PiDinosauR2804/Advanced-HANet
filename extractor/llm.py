from google import genai
from google.genai import types
import time
import re
import ast
from loguru import logger
from tqdm import tqdm

class Extractor():
    def __init__(self, api_key:str):
        """
        Initialize the Extractor with the provided API key and model.
        """
        self.api_key = api_key
    
    def extract_event(self, text:str, **args):
        """
        Extract events from text using Google Gemini API.
        """
        return [[{"text": text, 
                  "event_type": "meeting", 
                  "trigger_word": text.split()[0], 
                  "event_time": None, 
                  "event_location": None, 
                  "event_participants": []}]]  # Dummy response for the base class
        
class Extractor_Gemini(Extractor):
    def __init__(self, api_key:str):
        """
        Initialize the Extractor with the provided API key and model.
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.prompt = """You are an event extraction expert. Given a text, extract the event triggers. You should return a list of events in the last line with format:
The events are: [...]. 
Each event should be a dictionary with the following keys: "event_type", "trigger_word", "event_time", "event_location", "event_participants" and "description". 
The values for these keys should be extracted from the text. If any of the keys are not present in the text, return None for that key.
For example:
1. If the text is "John and Mary met at the park on Monday", the output should be:
The events are: [{{"event_type": "meeting", "trigger_word": "met", "event_time": "Monday", "event_location": "park", "event_participants": ["John", "Mary"], "description": "The trigger word met refers to the event where two or more parties encountered each other, marking the occurrence of a meeting or interaction"}}]
2. If the text is "The July 2006 earthquake was also centered in the Indian Ocean, from the coast of Java, and had a duration of more than three minutes.", the output should be:
The events are: [{{"event_type": "catastrophe", "trigger_word": "earthquake", "event_time": "July 2006", "event_location": "Indian Ocean", "event_participants": None, "description": "The trigger word earthquake refers to the event of the earth shaking, often causing destruction and damage"}}, 
                {{"event_type": "placing", "trigger_word": "centered", "event_time": "July 2006", "event_location": "Indian Ocean", "event_participants": None, "description": "The trigger word centered refers to the event of being located at a specific point or area"}}]
3. If the text does not contain any events, return an empty list.
The events are: []

Now, please extract the events from the following text:
{content}
"""

    def response_to_string(self, response, idx=0):
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

    def extract_response(self, text:str):
        match = re.search(r"The events are:\s*(\[.*\])", text, re.DOTALL)

        if match:
            events_str = match.group(1)
            try:
                events_list = ast.literal_eval(events_str)
                return events_list
            
            except ValueError as e:
                logger.error(f"[EXTRACT EVENT] Error parsing events: {e}")
                return None
        else:
            logger.error(f"[EXTRACT EVENT] No events found in the response.")
            return None
        
    def validate_event_list(self, event_list:list)->list:
        """
        Validate the event list to ensure it contains the required keys.
        """
        valid_events = []
        for event in event_list:
            if not isinstance(event, dict):
                logger.error(f"[EXTRACT EVENT] Invalid event format: {event}")
            else:
                for key, value in event.items():
                    if isinstance(value, str):
                        if len(value) == 0:
                            event[key] = None
                        else:
                            event[key] = value.lower().strip()
                            
                    elif isinstance(value, list):
                        if len(value) == 0:
                            event[key] = None
                        else:
                            event[key] = [v.lower().strip() for v in value if isinstance(v, str)]
                    else:
                        event[key] = None
            if event.get("trigger_word") is not None:
                valid_events.append(event)
            else:
                logger.error(f"[EXTRACT EVENT] Invalid event: {event}")
        if len(valid_events) == 0:
            logger.error(f"[EXTRACT EVENT] No valid events found in the response.")
            return None
        else: 
            # logger.info(f"[EXTRACT EVENT] Found {valid_events}")
            return valid_events

    def extract_event(self, text:str, model="gemini-2.0-flash", candidate=1):
        """
        Extract events from text using Google Gemini API.
        """
        # Gen answer
        response = self.client.models.generate_content(
            model=model,
            contents=self.prompt.format(content=text),
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                candidate_count=candidate
            )
        )

        res = []
        for idx in range(len(response.candidates)):
            response_string = self.response_to_string(response, idx)
            event_list = self.extract_response(response_string)
            if event_list:
                valid_event_list = self.validate_event_list(event_list)
                if valid_event_list:
                    res.append(valid_event_list)
                else:
                    logger.error(f"[EXTRACT EVENT] No valid events found in the response.")
            else:
                logger.error(f"[EXTRACT EVENT] No events found in the response.")

        return res
    
def is_quota_exhausted_error(e: Exception):
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)

def is_valid_extractor(extractor, text="australia won the tournament, beating pakistan in the final by 25 runs.", max_try=2):
    for _ in range(max_try):
        try:
            _ = extractor.extract_event(text, model="gemini-2.0-flash", candidate=1)
        except Exception as e:
            if is_quota_exhausted_error(e):
                return False
            time.sleep(5)
    return True