from flask import Flask, render_template, request, redirect, url_for
from google.cloud import vision, storage, vision_v1, speech
from google.cloud.vision_v1 import types
#from google.cloud.speech import enums, types
from decouple import config
import openai
import os
import requests
import logging
import json
from config import DevelopmentConfig, ProductionConfig, TestingConfig
from dotenv import load_dotenv

app = Flask(__name__)

# Set the configuration based on an environment variable
flask_env = os.environ.get('FLASK_ENV', 'development').lower()
load_dotenv()  # Take environment variables from .env.

if flask_env == 'production':
    app.config.from_object(ProductionConfig)
elif flask_env == 'testing':
    app.config.from_object(TestingConfig)
else:
    app.config.from_object(DevelopmentConfig)  # Default is development

logging.basicConfig(filename='error.log', level=logging.ERROR)

# Setup OpenAI's API Key
OPENAI_API_KEY = config('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Set up the Vision API client
SERVICE_ACCOUNT_PATH = config('GOOGLE_APPLICATION_CREDENTIALS')
vision_client = vision.ImageAnnotatorClient.from_service_account_json(SERVICE_ACCOUNT_PATH)
speech_client = speech.SpeechClient.from_service_account_json(SERVICE_ACCOUNT_PATH)


# Setup Google Cloud Storage client
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_PATH)
bucket_name = 'bagarab_nov2'
bucket = storage_client.bucket(bucket_name)

def extract_text_from_file(gcs_path, mime_type):
    if mime_type == 'application/pdf':
        response = vision_client.batch_annotate_files({
            'requests': [{
                'input_config': {
                    'gcs_source': {'uri': gcs_path},
                    'mime_type': 'application/pdf'
                },
                'features': [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
            }]
        })
        return response.responses[0].responses[0].full_text_annotation.text
    else:
        response = vision_client.document_text_detection(image=vision.Image(source=vision.ImageSource(image_uri=gcs_path)))
        return response.full_text_annotation.text

def async_detect_document(gcs_source_uri, mime_type):
    client = vision.ImageAnnotatorClient()

    # Define the source and the type of file
    gcs_source = types.GcsSource(uri=gcs_source_uri)
    input_config = types.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

    # Where to write the results
    output_uri = f'gs://{bucket_name}/output/'

    # The destination and output config
    gcs_destination = types.GcsDestination(uri=output_uri)
    output_config = types.OutputConfig(gcs_destination=gcs_destination, batch_size=20)

    # The async request
    async_request = types.AsyncAnnotateFileRequest(
        features=[types.Feature(type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        input_config=input_config,
        output_config=output_config
    )

    operation = client.async_batch_annotate_files(requests=[async_request])

    # Wait for the operation to complete
    result = operation.result(timeout=300)

    # Log the raw result for debugging
    logging.info(f"Raw result from Vision API: {result}")

    # Once the result is ready, fetch the output files from the GCS bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    prefix = output_uri[len('gs://')+len(bucket_name)+1:]  # Correctly parse the prefix
    blob_list = list(bucket.list_blobs(prefix=prefix))

    full_text = ''
    for blob in blob_list:
        if blob.name.endswith('/') and not blob.size:  # Skip directory placeholder
            continue
        json_string = blob.download_as_text()
        if not json_string.strip():  # Check if the blob is empty or whitespace
            logging.error(f"Blob {blob.name} is empty or whitespace.")
            continue

        try:
            # Parse the JSON response
            response = json.loads(json_string)
            for page_response in response['responses']:
                # Each page_response must be checked for the presence of fullTextAnnotation
                if 'fullTextAnnotation' in page_response:
                    full_text += page_response['fullTextAnnotation']['text']
        except json.JSONDecodeError as e:
            logging.error(f"An error occurred while parsing JSON from blob {blob.name}: {e}")
            continue
        except KeyError as e:
            logging.error(f"Missing expected field in the response: {e}")
            continue
        # Delete the blob after processing
        delete_blob(bucket_name, blob.name)

    return full_text

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

def summarize_text(text):
    """Use GPT-3.5-turbo to summarize the given text."""
    
    prepared_text, truncate_message = prepare_text(text)

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please summarize this text and keep it brief, but first, indicate what kind of text this is (e.g. article, essay, lab report, song). Make sure to give key takeaways."
                },
                {
                    "role": "user",
                    "content": prepared_text
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

        if 'choices' in response_json and len(response_json['choices']) > 0:
            # If the API call was successful and returned a summary, return it with the truncate message.
            return response_json['choices'][0]['message']['content'].strip(), truncate_message
        else:
            # If there's no 'choices' in the response, it's likely an error.
            error_message = response_json.get('error', {}).get('message', "Couldn't generate a summary.")
            print(f"OpenAI Response: {response_json}")
            return error_message, truncate_message

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return str(e), truncate_message
    
# Prepare text truncates the text if it is too long, and provides a message about the truncation.
def prepare_text(text, max_length=4096):
    """
    If text is longer than max_length, truncate it and provide a message.
    """
    truncate_message = ""
    if len(text) > max_length:
        truncated_amount = len(text) - max_length
        truncate_message = f"Text has been truncated by {truncated_amount} characters because it was too long."
        text = text[:max_length]
    return text, truncate_message

def upload_to_gcs(uploaded_file):
    blob = bucket.blob(uploaded_file.filename)
    blob.upload_from_file(uploaded_file, content_type=uploaded_file.content_type)
    return f"gs://{bucket_name}/{uploaded_file.filename}"

def transcribe_voice(voice_file):
    """
    Transcribe the given voice file using Google Cloud Speech-to-Text.
    """
def transcribe_voice(voice_file):
    audio_content = voice_file.read()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = speech_client.recognize(config=config, audio=audio)
    transcript = ""

    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript

def analyze_mood(text):
    """
    Analyze the mood of the given text using OpenAI's GPT-3.5 API.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the mood of the following statement: \"{text}\"",
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        return "Error in mood analysis."

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/summarizer')
def summarizer():
    return render_template('summarizer.html')

@app.route('/upload-file', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    try:
        if uploaded_file.content_type == 'application/pdf':
            gcs_uri = upload_to_gcs(uploaded_file)
            extracted_text = async_detect_document(gcs_uri, 'application/pdf')
            delete_blob(bucket_name, os.path.basename(gcs_uri))  # Cleanup after processing
            return render_template('output.html', extracted_text=extracted_text)
        else:
            # Handle other file types if necessary
            pass
    except Exception as e:
        logging.error(f"An error occurred during file upload or processing: {e}")
        return render_template("error.html", error=str(e)), 500    

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('original_text')
    summarized, truncate_message = summarize_text(text)
    summarized = summarized.replace('\n', '<br>')  # Convert newlines to HTML breaks

    # Pass the truncate_message to the template as well
    return render_template('output.html', extracted_text=text, summary=summarized, truncate_message=truncate_message)

@app.route('/voice-analyzer')
def voice_analyzer():
    return render_template('voice_analyzer.html')

@app.route('/analyze-voice', methods=['POST'])
def analyze_voice():
    if 'voice_file' in request.files:
        # Handling uploaded file
        voice_file = request.files['voice_file']
        transcribed_text = transcribe_voice(voice_file)
    else:
        # Handling recorded audio (assuming it's sent as a base64 encoded string)
        audio_data = request.form.get('audio_data')
        if audio_data:
            # Convert base64 to audio file
            voice_file = convert_base64_to_audio(audio_data)
            transcribed_text = transcribe_voice(voice_file)
        else:
            return "No audio data received", 400

    mood_analysis = analyze_mood(transcribed_text)

    return render_template('voice_analysis_result.html', transcribed_text=transcribed_text, mood_analysis=mood_analysis)

def convert_base64_to_audio(base64_string):
    import base64
    from io import BytesIO

    # Decode base64 string to bytes
    audio_bytes = base64.b64decode(base64_string)
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "recorded_voice.wav"  # You may need to adjust the file format
    return audio_file

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(e)
    return render_template("500.html"), 500

@app.errorhandler(404)
def not_found(e):
    return "404 Not Found: The requested URL was not found on the server.", 404

if __name__ == '__main__':
    app.run()