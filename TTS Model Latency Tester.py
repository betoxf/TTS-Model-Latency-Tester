import warnings
import time
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import io
from transformers import pipeline

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# List of TTS models to test
models = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/tacotron2-DDC_ph",
    "tts_models/en/vctk/fast_pitch",
    "tts_models/en/vctk/tacotron2-DDC",  # New model (check license)
    "tts_models/en/vctk/fast_pitch_v2",  # New model (check license)
    "tts_models/en/vctk/vits",           # New model (check license)
    "facebook/fastspeech2-en-ljspeech",  # Hugging Face model (check license)
    "facebook/fastspeech2-en-vctk"       # Hugging Face model (check license)
]

# Phrase to test
test_phrase = "Hello, how are you?"

# Function to test a single TTS model from TTS library and measure latency
def test_tts_model_tts_lib(model_name):
    """
    Test a TTS model from the TTS library by generating speech from text and measuring latency.

    Parameters:
    model_name (str): The name of the TTS model to test.

    Returns:
    float: The latency in seconds, or None if an error occurred.
    """
    try:
        # Initialize the TTS model
        tts = TTS(model_name=model_name)
        
        # Measure the time taken to generate speech
        start_time = time.time()
        tts.tts_to_file(text=test_phrase, file_path="output.wav")
        end_time = time.time()
        latency = end_time - start_time
        print(f"Model: {model_name} | Latency: {latency:.4f} seconds")

        # Play the generated audio to verify
        with open("output.wav", "rb") as audio_file:
            audio_stream = audio_file.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_stream), format="wav")
            play(audio)

        return latency
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return None

# Function to test a single TTS model from Hugging Face and measure latency
def test_tts_model_huggingface(model_name):
    """
    Test a TTS model from Hugging Face by generating speech from text and measuring latency.

    Parameters:
    model_name (str): The name of the TTS model to test.

    Returns:
    float: The latency in seconds, or None if an error occurred.
    """
    try:
        # Initialize the Hugging Face TTS model
        tts_pipeline = pipeline("text-to-speech", model=model_name)
        
        # Measure the time taken to generate speech
        start_time = time.time()
        tts_pipeline(test_phrase, output_dir="output")
        end_time = time.time()
        latency = end_time - start_time
        print(f"Model: {model_name} | Latency: {latency:.4f} seconds")

        # Play the generated audio to verify
        audio = AudioSegment.from_wav("output/tts_output.wav")
        play(audio)

        return latency
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return None

# Main function to test multiple TTS models and log the results
def main():
    """
    Main function to test multiple TTS models and log their latencies.
    """
    results = {}
    for model in models:
        if model.startswith("tts_models"):
            latency = test_tts_model_tts_lib(model)
        else:
            latency = test_tts_model_huggingface(model)
        
        if latency is not None:
            results[model] = latency

    # Print the results sorted by latency
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    print("\nLatency Results (sorted):")
    for model, latency in sorted_results:
        print(f"Model: {model} | Latency: {latency:.4f} seconds")

# Run the main function
if __name__ == "__main__":
    main()
