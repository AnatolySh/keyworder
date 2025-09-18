import warnings
warnings.filterwarnings("ignore")


import os
import sys
import base64
import json
import subprocess
import csv
import logging
import shutil
import argparse
import io
from concurrent.futures import ThreadPoolExecutor
import multiprocessing 

from PIL import Image, ExifTags
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, ValidationError

def get_application_path():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    return application_path

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = get_application_path()
    return os.path.join(base_path, relative_path)

def get_ffmpeg_executable_path():
    ffmpeg_resource_name = "ffmpeg.exe" 

    if getattr(sys, 'frozen', False): 
        bundled_ffmpeg_path = resource_path(ffmpeg_resource_name)
        if os.path.exists(bundled_ffmpeg_path):
            if not os.access(bundled_ffmpeg_path, os.X_OK): 
                try:
                    os.chmod(bundled_ffmpeg_path, 0o755) 
                    logging.info(f"Set execute permission for bundled ffmpeg: {bundled_ffmpeg_path}")
                except Exception as e:
                    logging.warning(f"Could not set execute permission for {bundled_ffmpeg_path}: {e}")
            return bundled_ffmpeg_path
        else:
            logging.error(f"Bundled ffmpeg not found at expected location: {bundled_ffmpeg_path}. Falling back to PATH.")
    return "ffmpeg"

SCRIPT_DIR = get_application_path()
SETTINGS_FILE_PATH = os.path.join(get_application_path(), "settings.json")

def load_app_settings(settings_path):
    try:
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)

        if "OPENAI_API_KEY" not in settings_data:
            logging.critical("OPENAI_API_KEY not found in settings.json. Please add it.")
            sys.exit(1)
        if not settings_data["OPENAI_API_KEY"] or not settings_data["OPENAI_API_KEY"].startswith("sk-"):
            api_key_display = settings_data.get("OPENAI_API_KEY", "")[:10] + "..." if settings_data.get("OPENAI_API_KEY") else "None"
            logging.warning(
                f"OPENAI_API_KEY in settings.json ('{api_key_display}') "
                "does not look like a valid key or is empty."
            )

        if "LLM_GENERATION_PROMPT" not in settings_data or not settings_data["LLM_GENERATION_PROMPT"]:
            logging.critical("LLM_GENERATION_PROMPT not found or is empty in settings.json. This is required.")
            sys.exit(1)

        settings_data.setdefault("MAX_WORKERS", 10)
        settings_data.setdefault("LOG_LEVEL", "INFO")
        settings_data.setdefault("DEFAULT_LOG_DIR", None) 
        settings_data.setdefault("IMAGE_EXTENSIONS", ['.jpg', '.jpeg', '.png', '.webp'])
        settings_data.setdefault("VIDEO_EXTENSIONS", ['.mp4', '.mov', '.avi', '.mkv'])
        settings_data.setdefault("OPENAI_MODEL", "gpt-4o-mini")
        settings_data.setdefault("STOP_WORDS", [
            "and", "or", "but", "with", "without", "on", "in", "at", "for", "to",
            "from", "by", "of", "a", "an", "the", "stock", "photo", "video",
            "vector", "3d", "render", "illustration", "rendering", "photography",
        ])
        return settings_data
    except FileNotFoundError:
        logging.critical(f"Settings file not found: {settings_path}. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"Error decoding JSON from {settings_path}. Please check its format.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while loading settings: {e}", exc_info=True)
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s (bootstrap)')


APP_SETTINGS = None
client = None

class Description(BaseModel):
    title: str
    keywords: list[str]

def setup_logging(log_file_path, level="INFO"):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level: {level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=numeric_level
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def fix_image_orientation(img):
    try:
        for orientation_tag_key in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation_tag_key] == 'Orientation':
                break
        
        exif = img._getexif()
        if exif is not None:
            orientation_val = exif.get(orientation_tag_key)
            if orientation_val == 3:
                img = img.rotate(180, expand=True)
            elif orientation_val == 6:
                img = img.rotate(270, expand=True)
            elif orientation_val == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return img

def resize_image_to_longest_side_and_get_base64(image_path, target_longest_side=512):
    try:
        with Image.open(image_path) as img:
            img = fix_image_orientation(img) 
            img = img.convert("RGB") 
            original_width, original_height = img.size

            if original_width == 0 or original_height == 0:
                logging.warning(f"Image has zero dimension: {image_path}. Skipping resize.")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()
                return base64.b64encode(img_bytes).decode("utf-8")

            if max(original_width, original_height) > target_longest_side:
                if original_width > original_height:
                    new_width = target_longest_side
                    new_height = int(original_height * target_longest_side / original_width)
                else: 
                    new_height = target_longest_side
                    new_width = int(original_width * target_longest_side / original_height)
                
                new_width = max(1, new_width)
                new_height = max(1, new_height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85) 
            img_bytes = buffer.getvalue()
            base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            
            final_width, final_height = Image.open(io.BytesIO(img_bytes)).size
            logging.debug(f"Processed image: {os.path.basename(image_path)}. Original: {original_width}x{original_height}, Final for API: {final_width}x{final_height}")
            return base64_encoded
    except Exception as e:
        logging.error(f"Failed to open, resize, or encode image {image_path}: {e}", exc_info=True)
        return None

def _run_ffmpeg_command(command_args_after_executable, description):
    ffmpeg_exe = get_ffmpeg_executable_path()
    full_command = [ffmpeg_exe] + command_args_after_executable
    try:
        logging.debug(f"Running FFmpeg command: {' '.join(full_command)}")
        process = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if process.returncode != 0:
            logging.error(f"FFmpeg failed for {description}. Return code: {process.returncode}")
            logging.error(f"FFmpeg stdout: {process.stdout.strip()}")
            logging.error(f"FFmpeg stderr: {process.stderr.strip()}")
            return False
        logging.info(f"FFmpeg successfully {description}.")
        return True
    except FileNotFoundError: 
        logging.error(f"FFmpeg executable ('{ffmpeg_exe}') not found while trying to {description}. Ensure it's bundled or in PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error during FFmpeg for {description}: {e}", exc_info=True)
        return False

def extract_first_and_last_frames(video_path, temp_dir, original_filename_base):
    first_frame_path = None
    last_frame_path = None
    first_frame_filename = f"{original_filename_base}_{os.urandom(4).hex()}_first_frame.jpg"
    first_frame_output_path = os.path.join(temp_dir, first_frame_filename)
    first_frame_setting = APP_SETTINGS.get("FIRST_FRAME", "")
    command_first_args = [
        "-ss", first_frame_setting, 
        "-i", video_path,
        "-vf", "select='eq(n\\,0)'",
        "-an", "-q:v", "2", 
        first_frame_output_path, "-y" 
    ]
    if _run_ffmpeg_command(command_first_args, f"extract first frame from {os.path.basename(video_path)} to {first_frame_filename}"):
        first_frame_path = first_frame_output_path
    
    last_frame_filename = f"{original_filename_base}_{os.urandom(4).hex()}_last_frame.jpg"
    last_frame_output_path = os.path.join(temp_dir, last_frame_filename)
    last_frame_setting = APP_SETTINGS.get("LAST_FRAME", "")
    command_last_args = [
        "-sseof", last_frame_setting, 
        "-i", video_path,
        "-an", "-frames:v", "1", 
        "-q:v", "2", 
        last_frame_output_path, "-y" 
    ]
    if _run_ffmpeg_command(command_last_args, f"extract last frame from {os.path.basename(video_path)} to {last_frame_filename}"):
        last_frame_path = last_frame_output_path
    return first_frame_path, last_frame_path

def generate_metadata(image_base64_list, llm_prompt_text):
    try:
        openai_model = APP_SETTINGS.get("OPENAI_MODEL") 
        stop_words_list = APP_SETTINGS.get("STOP_WORDS", [])

        if not llm_prompt_text: 
            logging.error("LLM_GENERATION_PROMPT is missing or empty. Cannot generate metadata.")
            return None, None
        if not image_base64_list:
            logging.error("No image data provided to generate_metadata.")
            return None, None

        content_parts = [{"type": "text", "text": llm_prompt_text}]
        for b64_image in image_base64_list:
            if b64_image: 
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"}}
                )
        
        if len(content_parts) == 1: 
            logging.error("No valid image data could be prepared for API request.")
            return None, None

        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": content_parts}],
            response_format={"type": "json_object"},
        )

        if response and response.choices and len(response.choices) > 0:
            content_str = response.choices[0].message.content
            try:
                parsed_data = json.loads(content_str)
                description_obj = Description(**parsed_data)
                title = description_obj.title.strip()
                raw_keywords = description_obj.keywords
                seen = set()
                cleaned_keywords_list = []
                for kw in raw_keywords:
                    kw_lower = str(kw).strip().lower()
                    if kw_lower and kw_lower not in stop_words_list and kw_lower not in seen:
                        seen.add(kw_lower)
                        cleaned_keywords_list.append(kw_lower)
                keywords_str = ",".join(cleaned_keywords_list)
                if not title or not keywords_str:
                    logging.warning(f"Generated title or keywords are empty after cleaning. Title: '{title}', Keywords: '{keywords_str}'")
                return title, keywords_str
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON response from OpenAI: {e}. Response content: {content_str}")
            except ValidationError as e:
                logging.error(f"OpenAI response failed Pydantic validation: {e}. Response content: {content_str}")
            except Exception as e:
                logging.error(f"Error processing OpenAI response: {e}. Response content: {content_str}", exc_info=True)
        else:
            logging.error("No valid response or choices from OpenAI API.")
        return None, None 
    except Exception as e:
        logging.error(f"OpenAI API request failed: {e}", exc_info=True)
        return None, None

def write_csv(file_path, headers, data):
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        logging.info(f"Successfully wrote CSV file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to write CSV file {file_path}: {e}", exc_info=True)

def process_file(file_info, image_extensions, video_extensions, input_root_dir):
    file_path, relative_file_path = file_info
    ext = os.path.splitext(file_path)[1].lower()
    temp_files_to_delete = []
    original_filename = os.path.basename(file_path)
    original_filename_base = os.path.splitext(original_filename)[0]
    temp_file_dir = os.path.dirname(file_path) 
    final_title_str = None
    final_keywords_str = "" 
    base_llm_prompt = APP_SETTINGS.get("LLM_GENERATION_PROMPT")
    if not base_llm_prompt:
        logging.error("LLM_GENERATION_PROMPT is missing from settings. Cannot process file.")
        return None

    try:
        if ext in video_extensions:
            logging.info(f"Processing video: {original_filename}")
            frame1_path, frame2_path = extract_first_and_last_frames(file_path, temp_file_dir, original_filename_base)
            
            video_frames_b64_list = []
            b64_frame1_for_retry = None

            if frame1_path:
                temp_files_to_delete.append(frame1_path)
                b64_f1 = resize_image_to_longest_side_and_get_base64(frame1_path, 512)
                if b64_f1:
                    video_frames_b64_list.append(b64_f1)
                    b64_frame1_for_retry = b64_f1 
            
            if frame2_path:
                temp_files_to_delete.append(frame2_path)
                b64_f2 = resize_image_to_longest_side_and_get_base64(frame2_path, 512)
                if b64_f2:
                    video_frames_b64_list.append(b64_f2)
            
            if not video_frames_b64_list: 
                logging.error(f"No frames could be extracted/resized/encoded for video {original_filename}.")
                errors_folder_base = os.path.join(input_root_dir, "errors")
                os.makedirs(errors_folder_base, exist_ok=True)
                error_path_destination = os.path.join(errors_folder_base, relative_file_path)
                os.makedirs(os.path.dirname(error_path_destination), exist_ok=True)
                try:
                    shutil.move(file_path, error_path_destination)
                except Exception as e_move:
                    logging.error(f"Failed to move {original_filename} to errors (no frames): {e_move}", exc_info=True)
                return None

            num_frames = len(video_frames_b64_list)
            num_frames_text = "frame" if num_frames == 1 else f"{num_frames} frames (first and last if two are provided)"
            video_llm_prompt = f"These are {num_frames_text} from a video. " + base_llm_prompt
            
            logging.info(f"Generating metadata for video {original_filename} using {num_frames} frame(s).")
            title, keywords = generate_metadata(video_frames_b64_list, video_llm_prompt)

            if title is None and keywords is None: 
                logging.info(f"Initial metadata generation failed for video {original_filename}. Retrying with first frame only.")
                if b64_frame1_for_retry: 
                    retry_prompt = f"This is the first frame of a video. " + base_llm_prompt
                    title, keywords = generate_metadata([b64_frame1_for_retry], retry_prompt)
                    if keywords:
                         logging.info(f"Retry with first frame successful for video {original_filename}.")
                    else:
                        logging.warning(f"Retry with first frame also yielded no metadata for video {original_filename}.")
                else:
                    logging.warning(f"Cannot retry video {original_filename} as its first frame was not successfully processed initially.")
            
            final_title_str = title
            final_keywords_str = keywords if keywords is not None else ""
            
            if not final_keywords_str: 
                logging.error(f"No keywords generated for video {original_filename} after all attempts.")
                errors_folder_base = os.path.join(input_root_dir, "errors")
                os.makedirs(errors_folder_base, exist_ok=True)
                error_path_destination = os.path.join(errors_folder_base, relative_file_path)
                os.makedirs(os.path.dirname(error_path_destination), exist_ok=True)
                try:
                    shutil.move(file_path, error_path_destination)
                    logging.info(f"Moved {original_filename} to errors folder due to no keyword generation.")
                except Exception as e_move:
                    logging.error(f"Failed to move {original_filename} to errors folder: {e_move}. File remains at {file_path}", exc_info=True)
                return None 

        elif ext in image_extensions:
            logging.info(f"Processing image: {original_filename}")
            image_llm_prompt = base_llm_prompt
            
            image_base64 = resize_image_to_longest_side_and_get_base64(file_path, 512)
            
            if not image_base64: 
                logging.error(f"Could not get base64 for image {original_filename}. Moving to errors.")
                errors_folder_base = os.path.join(input_root_dir, "errors")
                os.makedirs(errors_folder_base, exist_ok=True)
                error_path_destination = os.path.join(errors_folder_base, relative_file_path)
                os.makedirs(os.path.dirname(error_path_destination), exist_ok=True)
                try:
                    shutil.move(file_path, error_path_destination)
                except Exception as e_move:
                    logging.error(f"Failed to move {original_filename} to errors (base64 fail): {e_move}", exc_info=True)
                return None

            title, keywords = generate_metadata([image_base64], image_llm_prompt)

            if title is None and keywords is None:
                logging.info(f"Retrying metadata generation for image: {original_filename}")
                title, keywords = generate_metadata([image_base64], image_llm_prompt)
                if title is None and keywords is None:
                    errors_folder_base = os.path.join(input_root_dir, "errors")
                    os.makedirs(errors_folder_base, exist_ok=True)
                    error_path_destination = os.path.join(errors_folder_base, relative_file_path)
                    os.makedirs(os.path.dirname(error_path_destination), exist_ok=True)
                    try:
                        shutil.move(file_path, error_path_destination)
                        logging.error(f"Metadata generation failed twice for image {original_filename}. Moved to {error_path_destination}")
                    except Exception as e_move:
                        logging.error(f"Failed to move {original_filename} to errors folder: {e_move}. File remains at {file_path}", exc_info=True)
                    return None
            final_title_str = title
            final_keywords_str = keywords if keywords is not None else ""
        else:
            logging.warning(f"Unsupported file type: {file_path}")
            return None
        
        final_title_str = final_title_str or "Error: No Title Generated" 
        return original_filename, final_title_str, final_keywords_str
    finally:
        for temp_path in temp_files_to_delete:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logging.error(f"Failed to remove temporary file {temp_path}: {e}", exc_info=True)

def main():
    global APP_SETTINGS, client

    APP_SETTINGS = load_app_settings(SETTINGS_FILE_PATH)
    try:
        if not APP_SETTINGS.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is empty in settings.")
        client = OpenAI(api_key=APP_SETTINGS["OPENAI_API_KEY"])
        logging.info(f"OpenAI client initialized successfully for model {APP_SETTINGS.get('OPENAI_MODEL')}.")
    except ValueError as ve:
        logging.critical(f"Failed to initialize OpenAI client: {ve}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        sys.exit(1)


    parser = argparse.ArgumentParser(description="Process images and videos to generate metadata and CSV files.")
    default_input_dir = os.path.join(SCRIPT_DIR, "input")
    default_output_dir = os.path.join(SCRIPT_DIR, "output")
    log_dir_from_settings = APP_SETTINGS.get("DEFAULT_LOG_DIR")
    if log_dir_from_settings and os.path.isabs(log_dir_from_settings):
        effective_default_log_dir = log_dir_from_settings
    elif log_dir_from_settings: 
        effective_default_log_dir = os.path.join(SCRIPT_DIR, log_dir_from_settings)
    else: 
        effective_default_log_dir = os.path.join(SCRIPT_DIR, "logs")

    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                        help=f"Directory containing image/video files. Default: '{default_input_dir}'")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help=f"Directory to store output CSV files. Default: '{default_output_dir}'")
    parser.add_argument("--log_dir", type=str, default=effective_default_log_dir,
                        help=f"Directory to store the log file. Default: '{effective_default_log_dir}'.")
    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_dir)
    log_directory = os.path.abspath(args.log_dir)
    output_directory = os.path.abspath(args.output_dir)

    for dir_path, dir_name in [(log_directory, "Log"), (output_directory, "Output")]:
        if not os.path.isdir(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                print(f"Error: Could not create {dir_name.lower()} directory '{dir_path}': {e}", file=sys.stderr)
                logging.critical(f"Could not create {dir_name.lower()} directory '{dir_path}': {e}")
                sys.exit(1)
    
    log_file_level = APP_SETTINGS.get("LOG_LEVEL", "INFO").upper()
    log_file_path = os.path.join(log_directory, "processing.log")
    setup_logging(log_file_path, level=log_file_level)

    if not os.path.isdir(input_directory):
        logging.critical(f"Input directory '{input_directory}' not found. Please create it or specify a valid directory.")
        sys.exit(1)

    logging.info("Script started with settings loaded.")
    logging.info(f"Application Base Path (SCRIPT_DIR): {SCRIPT_DIR}")
    logging.info(f"Settings File Path: {SETTINGS_FILE_PATH}")
    logging.info(f"FFmpeg Executable Path: {get_ffmpeg_executable_path()}")
    logging.info(f"Using OpenAI model: {APP_SETTINGS.get('OPENAI_MODEL')}")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output directory: {output_directory}")
    logging.info(f"Log file: {log_file_path} (Level: {log_file_level})")
    logging.info(f"Stop words loaded: {APP_SETTINGS.get('STOP_WORDS')}")

    image_extensions = APP_SETTINGS.get("IMAGE_EXTENSIONS")
    video_extensions = APP_SETTINGS.get("VIDEO_EXTENSIONS")
    logging.info(f"Supported image extensions: {image_extensions}")
    logging.info(f"Supported video extensions: {video_extensions}")

    all_files_to_process = []
    for root, dirs, files in os.walk(input_directory):
        files = [f for f in files if not f.startswith('.')]
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if os.path.basename(root) == "errors" and os.path.abspath(os.path.dirname(root)) == os.path.abspath(input_directory):
            logging.info(f"Skipping 'errors' directory at the root of input: {root}")
            dirs[:] = []
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions + video_extensions:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_directory)
                all_files_to_process.append((full_path, relative_path))

    if not all_files_to_process:
        logging.info(f"No supported image or video files found in '{input_directory}'. Exiting.")
        sys.exit(0)
    else:
        logging.info(f"Found {len(all_files_to_process)} files to process.")

    adobestock_data = []
    stocksubmitter_data = []
    max_workers = int(APP_SETTINGS.get("MAX_WORKERS"))
    logging.info(f"Using {max_workers} worker threads for processing.")
    description_ext_setting = APP_SETTINGS.get("DESCRIPTION_EXT", "")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_info, image_extensions, video_extensions, input_directory)
                   for file_info in all_files_to_process]
        for future in tqdm(futures, desc="Processing files", total=len(all_files_to_process)):
            try:
                result = future.result()
                if result:
                    filename, title, keywords = result
                    title = title if title is not None else "Error: No Title Generated"
                    keywords = keywords if keywords is not None else "Error: No Keywords Generated"
                    adobestock_data.append([filename, title, keywords, "", ""])
                    stocksubmitter_data.append([filename, title, title + description_ext_setting, keywords, ""])
                    stocksubmitter_data[-1] = [col.replace("..", ".") for col in stocksubmitter_data[-1]]
            except Exception as e:
                logging.error(f"Unhandled error processing a file future: {e}", exc_info=True)

    adobestock_headers = ["Filename", "Title", "Keywords", "Category", "Releases"]
    stocksubmitter_headers = ["Filename", "Title", "Description", "Keywords", "Release name"]
    adobestock_csv = os.path.join(output_directory, "adobestock.csv")
    stocksubmitter_csv = os.path.join(output_directory, "stocksubmitter.csv")
    write_csv(adobestock_csv, adobestock_headers, adobestock_data)
    write_csv(stocksubmitter_csv, stocksubmitter_headers, stocksubmitter_data)
    logging.info(f"Processing complete. CSV files generated in '{output_directory}'.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()