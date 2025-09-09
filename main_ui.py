import gradio as gr
from gradio import ChatMessage
import time
import base64
import wave
import pyaudio
import keyboard
from google.cloud import speech
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

from main_client import MedicalChatbot

medbot = MedicalChatbot()

img_ = None
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
TEMP_WAV_FILE = "temp_audio.wav"

# Global language parameter
LANGUAGE_CODE = "en-US"  # Default language

# Set safety settings
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# vertexai.init(project="psychic-heading-443019-m5", location="us-central1")
# model = GenerativeModel("gemini-1.5-flash-002")

def record_audio():

    audio = pyaudio.PyAudio()

    print("Recording...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not keyboard.is_pressed('space'):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(TEMP_WAV_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return TEMP_WAV_FILE


# Function to convert audio to text using Google Cloud Speech-to-Text API
def speech_to_text(config: speech.RecognitionConfig, audio_file: str) -> speech.RecognizeResponse:

    client = speech.SpeechClient()

    with open(audio_file, "rb") as audio:
        content = audio.read()

    audio = speech.RecognitionAudio(content=content)
    response = client.recognize(config=config, audio=audio)
    return response


def get_audio_response(response: speech.RecognizeResponse):
    """
    Extracts and returns the transcript from the speech-to-text response.
    """
    for result in response.results:
        best_alternative = result.alternatives[0]
        print("-" * 80)
        print(f"Transcript: {best_alternative.transcript}")
        return best_alternative.transcript
    return ""


def generate_audio_response(history):
    # Record the user's speech input
    audio_file = record_audio()  # This will record audio until spacebar is pressed
    print(f"Audio file saved at {audio_file}")
    
    # Convert audio to text using speech-to-text
    config = speech.RecognitionConfig(
        language_code=LANGUAGE_CODE,  # Use the global language code
        enable_automatic_punctuation=True,
    )
    response = speech_to_text(config, audio_file)
    user_input = get_audio_response(response)  # Get the transcript
    
    # Add the user's text query to history
    history.append(ChatMessage(role="user", content=user_input))
    yield history

    messages_as_dict = [{"role": msg.role, "content": msg.content} for msg in history]
    assistant_response = generate(user_input, messages_as_dict[1:])
    # assistant_response = " ".join(responses) if responses else "No response generated."
    
    print(assistant_response)
    history.append(ChatMessage(role="assistant", content=assistant_response))
    yield history


def translate_text(text: str, target_language: str) -> dict:
    """Translates text from English to Hindi."""
    from google.cloud import translate_v2 as translate

    # Initialize the Google Cloud Translate API client
    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Translate the text
    result = translate_client.translate(text, target_language=target_language)

    # Print the results
    # print("Original Text: {}".format(result["input"]))
    # print("Translated Text: {}".format(result["translatedText"]))
    # print("Detected Source Language: {}".format(result["detectedSourceLanguage"]))

    return result

def generate(input_text, message_as_dict, is_image=False):
    # If base language not english translate to english
    # print(message_as_dict)

    target_language = "en"
    if LANGUAGE_CODE != "en-US" and is_image == False:
        translate = translate_text(input_text, "en")
        input_text = translate["translatedText"]
        target_language = translate["detectedSourceLanguage"]
        print("Translated: ", input_text)


    # responses = model.generate_content(
    #     [input_text],
    #     generation_config=generation_config,
    #     safety_settings=safety_settings,
    #     stream=True,
    # )

    # res = []
    # for response in responses:
    #     res.append(response.text)
    
    # assistant_response = " ".join(res) if res else "No response generated."

    is_first = True
    if len(message_as_dict) > 1:
        is_first = False
    
    # global img_

    assistant_response = None
    if img_ != None:
        assistant_response = medbot.run_combined_rag_pipeline(message_as_dict[1:], image_data=img_, is_first_query=True, medical_history=patient_history)
        # img_ = None
    else:
        assistant_response = medbot.run_combined_rag_pipeline(message_as_dict, image_data=None, is_first_query=True, medical_history=patient_history)
    
    if LANGUAGE_CODE != "en-US" and is_image == False:
        assistant_response = translate_text(assistant_response, target_language)["translatedText"]
        print("Translated output: ", assistant_response)
    return assistant_response

    # Translate the response back to chosen language


def handle_chat_input(message, history):
    history.append(ChatMessage(role="user", content=message))
    try:
        messages_as_dict = [{"role": msg.role, "content": msg.content} for msg in history]
        assistant_response = generate(message, messages_as_dict[1:])
        # assistant_response = " ".join(responses) if responses else "No response generated."
    except Exception as e:
        assistant_response = f"Error while generating response: {str(e)}"
    
    history.append(ChatMessage(role="assistant", content=assistant_response))
    return history


# Function to update the global LANGUAGE_CODE based on dropdown selection
def set_language(language_code):
    global LANGUAGE_CODE
    LANGUAGE_CODE = language_code
    print(f"Language set to: {LANGUAGE_CODE}")


def like(evt: gr.LikeData):
    print("User liked the response")
    print(evt.index, evt.liked, evt.value)


def handle_image_upload(image, history):
    """Handles the image upload and appends it to the chat history."""
    global img_ 

    if image:
        with open(image.name, "rb") as f:
            img_ = f.read() 
            encoded_image = base64.b64encode(img_).decode("utf-8")
        image_tag = f'<img src="data:image/png;base64,{encoded_image}" alt="uploaded image" style="max-width: 300px; max-height: 300px;" />'
    
        # try:
        #     messages_as_dict = [{"role": msg.role, "content": msg.content} for msg in history]
        #     assistant_response = generate(img_, messages_as_dict, is_image=True)
        # except Exception as e:
        #     assistant_response = f"Error while generating response: {str(e)}"
        
        history.append(ChatMessage(role="user", content=image_tag))
        # history.append(ChatMessage(role="assistant", content=assistant_response))
    return history




# Global variable to store patient history
patient_history = ""

def update_patient_history(history_input):
    global patient_history
    patient_history = history_input
    # print(f"Patient history updated: {patient_history}")
    return f"Patient history recorded: {len(patient_history.split())} words."

with gr.Blocks(css="""
    .send-button {
        background-color: #007BFF; /* Primary blue color */
        color: white;
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    .send-button:hover {
        background-color: #0056b3; /* Slightly darker blue for hover */
        transform: translateY(-2px); /* Lift effect */
    }

    .send-button:active {
        background-color: #004085; /* Darker blue for active state */
        transform: translateY(0); /* Reset lift effect */
    }

    .gr-input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }

    .gr-row {
        margin-bottom: 15px;
    }

    # .medico-chatbot {
    #     border: 2px solid #007BFF;
    #     border-radius: 15px;
    #     padding: 10px;
    # }
               
    .medico-chatbot {
        border: 2px solid #007BFF;
        border-radius: 15px;
        padding: 10px;
        # background-image: url('https://cdn0.iconfinder.com/data/icons/chatter-bot/64/15_Robot_bubble_chat_bot_message_support_virtual_assistant_health_doctor_hospital-512.png');
        background-size: auto 40%; /* Scale to max height while maintaining aspect ratio */
        background-position: center;
        background-repeat: no-repeat; /* Prevent tiling of the image */
        color: white; /* Ensure text is visible against the background */
    }

               
    .upload-box {
        background-color: #f9f9f9; /* Light grey background */
        border: 2px dashed #007BFF; /* Dashed border with theme color */
        border-radius: 12px;
        padding: 10px;
        font-size: 14px;
        color: #007BFF;
        text-align: center;
        width: 700px;
        height: 200px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .upload-box:hover {
        background-color: #e6f4ff; /* Light blue background on hover */
    }
               
    .gr-input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 14px;
        font-family: "Arial", sans-serif;
        background-color: #fdfdfe; /* Subtle off-white */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Light shadow for depth */
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .gr-input:focus {
        border-color: #007BFF; /* Highlighted border on focus */
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.2); /* Slight glow on focus */
    }
    
    .dropdown-container {
        background-color: #f7f9fc; /* Matches the theme */
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        font-family: "Arial", sans-serif;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .dropdown-container:hover {
        border-color: #007BFF;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
    }
    
    .chatbot-name {
        font-size: 24px;
        font-weight: bold;
        color: #007BFF;
        text-align: center;
        margin-bottom: 20px;
        font-family: "Arial", sans-serif;
        text-transform: uppercase;
    }

""") as demo:
    # with gr.Row():
    #     gr.Markdown("<div class='chatbot-name'>Medico - Your Medical Assistant</div>")  # Styled chatbot name
            
    with gr.Row():
        chatbot = gr.Chatbot(
            value=[ChatMessage(role="assistant", content="Hello, I am MedAId. Please type in your question.")],
            type="messages",
            height=500,
            show_copy_button=True,
            elem_classes="medico-chatbot"  # Custom class
        )

    with gr.Row():
        with gr.Column():
            language_dropdown = gr.Dropdown(
                label="Select Language",
                choices=["en-US", "hi-IN", "pt-PT"],
                value="en-US",
                type="value",
                elem_classes="dropdown-container"
            )
            chat_input = gr.Textbox(label="Your Message", placeholder="Type a message...", lines=1, elem_classes="gr-input")
            history_input = gr.Textbox(
                label="Patient History",
                placeholder="Enter patient history here...",
                lines=5,
                interactive=True,
                elem_classes="gr-input"  # Custom class
            )
        with gr.Column():
            send_button = gr.Button("Send", elem_classes="send-button")  # Custom class
            action_button = gr.Button("Record", elem_classes="send-button")  # Same class for uniform styling
            image_upload = gr.File(label="Upload Image", file_types=["image"], elem_classes="upload-box")

    state = gr.State([
        ChatMessage(role="assistant", content="Hello, please type in your question.")
    ])

    # Event bindings
    send_button.click(handle_chat_input, [chat_input, state], chatbot)
    send_button.click(lambda: "", None, chat_input)
    action_button.click(generate_audio_response, state, chatbot)
    image_upload.change(handle_image_upload, [image_upload, state], chatbot)
    chatbot.like(like)
    language_dropdown.change(set_language, [language_dropdown], None)
    
    # Bind patient history input box
    history_input.change(update_patient_history, [history_input], None)

if __name__ == "__main__":
    demo.launch()
