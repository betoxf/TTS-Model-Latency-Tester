TTS Model Latency Tester

This project tests multiple Text-to-Speech (TTS) models for their latency in converting text to speech. It measures the time taken by each model to generate speech from a given text and plays the generated audio for verification.

Features

Tests multiple TTS models.
Measures and logs the latency for each model.
Plays the generated audio for verification.
Outputs the latency results sorted by time taken.
Installation

Prerequisites
Python 3.6 or higher
ffmpeg (required by pydub for audio processing)
Installing ffmpeg
For macOS:

sh
Copy code
brew install ffmpeg
For other operating systems, please refer to the official FFmpeg installation guide.

Python Packages
Install the required Python packages using pip:

sh
Copy code
pip install TTS pydub
Usage

Clone the repository:
sh
Copy code
git clone https://github.com/your-username/tts-model-latency-tester.git
cd tts-model-latency-tester
Ensure you have installed all the prerequisites and Python packages.

Run the script:

sh
Copy code
python tts_latency_tester.py
Script Overview
The script tests multiple TTS models listed in the models list for their latency in converting text to speech.

Function test_tts_model

Initializes the TTS model.
Generates speech audio from the given text.
Measures the latency.
Plays the generated audio for verification.
Main Function

Iterates over the list of models.
Tests each model and logs the latencies.
Prints the latency results sorted by time taken.
Adding or Changing Models
To test different TTS models, modify the models list in the tts_latency_tester.py script with the desired model names.

Example Code
python
Copy code
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

# Phrase to test (you can use short text here for measuring short text or use a long one to test the long ones, models have different variations)
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
Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For questions or suggestions, please contact betoxf18@gmail.com
