import os
import pytesseract
import cv2
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from ytdl import download_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_frame_index(filename):
    frame_index = os.path.splitext(filename)[0].split("_")[-1]
    return frame_index


"""def add_frame_metadata(frame, frame_timestamp):
    # Add frame timestamp as metadata to the frame image
    frame_with_metadata = frame.copy()
    cv2.putText(frame_with_metadata, f"Timestamp: {frame_timestamp} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame_with_metadata
"""


"""def extract_frame_timestamp(frame_with_metadata):
    # Load the frame image
    frame_image = cv2.imread(frame_with_metadata)

    # Crop the region containing the timestamp
    timestamp_region = frame_image[30:60,
                       10:200]  # Adjust coordinates as needed

    # Convert the region to grayscale
    gray_timestamp_region = cv2.cvtColor(timestamp_region, cv2.COLOR_BGR2GRAY)

    # Apply OCR to the grayscale region
    timestamp_text = pytesseract.image_to_string(gray_timestamp_region,
                                                 config='--psm 6')

    # Convert the extracted text to an integer (assuming it contains only digits)
    frame_timestamp = int(timestamp_text)
    return frame_timestamp"""


def convert_video_to_frames(video_path, output_dir, capture_rate=0.5):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    os.makedirs(output_dir, exist_ok=True)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % int(frame_rate / capture_rate) == 0:
            # frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            # frame_with_metadata = add_frame_metadata(frame, frame_timestamp)

            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            # cv2.imwrite(frame_path, frame_with_metadata)

        frame_count += 1

    video.release()


def generate_captions(video_link, results):
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(
        "cuda")

    frame_output_dir = "frames"
    video_path = download_video(video_link)
    convert_video_to_frames(video_path, frame_output_dir, capture_rate=1)
    captions = ""
    for filename in sorted(os.listdir(frame_output_dir)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(frame_output_dir, filename)
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to("cuda",
                                                              torch.float16)
            out = model.generate(**inputs)
            generated_caption = processor.decode(out[0],
                                                 skip_special_tokens=True)  # Skip special tokens
            image.close()
            os.remove(frame_path)
            # frame_timestamp = extract_frame_timestamp(frame_path)  # Extract frame timestamp from metadata
            # caption_with_timestamp = f"Frame Timestamp: {frame_timestamp} ms, Caption: {generated_caption}"  # Add timestamp to the caption
            captions = captions + "\n"   + generated_caption #caption_with_timestamp
    print(captions)
    # Delete frames and downloaded video
    os.remove(video_path)  # Delete downloaded video
    torch.cuda.empty_cache()
    results.append(captions)
