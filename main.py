# imports
import os

from dotenv import find_dotenv, load_dotenv
from hf_helpers import print_gpu_utilization
from transcription import transcribe
from captions import generate_captions

if __name__ == "__main__":
    """
    if len(sys.argv) < 2:
        print("Please provide a YouTube video link as a command line argument.")
        sys.exit(1)

    video_link = sys.argv[1]
    """

    # Loading env variables
    _ = load_dotenv(find_dotenv())
    huggingface_api_key = os.getenv('HFACE_API_KEY')

    # Get the parent directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the directory for storing the models within the virtual environment
    models_dir = os.path.join(script_dir, "model_dir")

    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    speech_model_identifier = "openai/whisper-base"
    img_model_identifier = "microsoft/git-base"

    # transcribe the video
    results = []
    link = "https://youtu.be/ORMx45xqWkA"   #
    transcribe(link, results)
    generate_captions(link, results)





