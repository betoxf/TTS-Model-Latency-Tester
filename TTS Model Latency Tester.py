import warnings
import time
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import io

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# List of TTS models to test
models = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/tacotron2-DDC_ph",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/ljspeech/vits--neon",
    "tts_models/en/ljspeech/fast_pitch",
    "tts_models/en/ljspeech/overflow",
    "tts_models/en/ljspeech/neural_hmm",
    "tts_models/en/vctk/vits",
    "tts_models/en/vctk/fast_pitch"
]

# Phrase to test
test_phrase = "Hello, how are you?"

# Function to test a single TTS model and measure latency
def test_tts_model(model_name):
    """
    Test a TTS model by generating speech from text and measuring latency.

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

# Main function to test multiple TTS models and log the results
def main():
    """
    Main function to test multiple TTS models and log their latencies.
    """
    results = {}
    for model in models:
        latency = test_tts_model(model)
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
