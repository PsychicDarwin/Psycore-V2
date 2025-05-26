import ffmpeg, tempfile, os

def chunk_audio_size(file: str, chunk_size_mb: int) -> list:
    """
    Splits an audio file into chunks of a specified size.

    Args:
        file (str): Path to the audio file.
        chunk_size_mb (int): Size of each chunk in megabytes.

    Returns:
        list: List of tuples containing start and end times of each chunk in seconds.
    """
    probe = ffmpeg.probe(file)
    duration = float(probe['format']['duration'])
    chunk_size = chunk_size_mb * 1024 * 1024  
    bitrate = int(probe['format']['bit_rate']) / 8  
    chunk_duration = chunk_size / bitrate  

    chunks = []
    for start in range(0, int(duration), int(chunk_duration)):
        end = min(start + chunk_duration, duration)
        chunks.append((start, end))

    return chunks

def split_audio_file(file: str, chunk_size_mb: int) -> list:
    """
    Splits an audio file into smaller chunks and saves them to temporary files.
    Args:
        file (str): Path to the audio file.
        chunk_size_mb (int): Size of each chunk in megabytes.
    Returns:   
        list: List of paths to the temporary chunk files in order of sequence.
    """

    chunks = chunk_audio_size(file, chunk_size_mb)
    chunk_files = []

    for i, (start, end) in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.close()  
            output_file = temp_file.name
            ffmpeg.input(file, ss=start, to=end).output(output_file).run(quiet=True)
            chunk_files.append(output_file)

    return chunk_files

def clean_temp_files(files: list):
    """
    Cleans up temporary files created during audio processing.
    Args:
        files (list): List of paths to the temporary files to be deleted.
    """
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {str(e)}")