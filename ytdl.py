import yt_dlp


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]',
        'outtmpl': '%(id)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            video_path = f"{info_dict['id']}.mp4"
            return video_path
        except yt_dlp.DownloadError as e:
            print("Error during download:", e)
        except yt_dlp.ExtractorError as e:
            print("Error during extraction:", e)
        except yt_dlp.UnsupportedError as e:
            print("Unsupported format or video source:", e)
        except Exception as e:
            print("An unexpected error occurred:", e)


def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            audio_path = f"{info_dict['id']}.wav"
            return audio_path
        except yt_dlp.DownloadError as e:
            print("Error during download:", e)
        except yt_dlp.ExtractorError as e:
            print("Error during extraction:", e)
        except yt_dlp.UnsupportedError as e:
            print("Unsupported format or video source:", e)
        except yt_dlp.PostProcessingError as e:
            print("Error during post-processing:", e)
        except Exception as e:
            print("An unexpected error occurred:", e)
