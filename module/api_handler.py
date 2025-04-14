import time
import io
import base64
from typing import List, Optional, Dict, Any, Union, Tuple
from google import genai
from google.genai import types
from mistralai import Mistral
from openai import OpenAI
from rich_pixels import Pixels
from rich.progress import Progress
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from PIL import Image
from pathlib import Path
import re
from datetime import datetime

console = Console()

# Add at top of file with other globals
api_call_log = []

def log_api_call(model: str, status: str, elapsed: float, error: str = None):
    """Log API call details"""
    timestamp = datetime.now()
    api_call_log.append({
        'timestamp': timestamp,
        'model': model,
        'status': status,
        'elapsed': elapsed,
        'error': error
    })
    # Print immediate feedback
    console.print(f"[blue]{timestamp.strftime('%H:%M:%S.%f')[:-3]}[/blue] API call to [green]{model}[/green]: {status} ({elapsed:.2f}s){' - ' + error if error else ''}")
    
    # Show rolling window stats
    window_start = timestamp.timestamp() - 60
    window_calls = sum(1 for call in api_call_log if call['timestamp'].timestamp() > window_start)
    console.print(f"[yellow]Calls in last 60s: {window_calls}[/yellow]")

def api_process_batch(
    uri: str,
    mime: str,
    config,
    args,
    sha256hash: str,
) -> str:
    console.print("[magenta]DEBUG: ***** ENTERED api_process_batch *****[/magenta]")
    try:
        console.print("[magenta]DEBUG: Accessing initial prompts...[/magenta]")
        system_prompt = config["prompts"]["system_prompt"]
        prompt = config["prompts"]["prompt"]
        console.print("[magenta]DEBUG: Initial prompts accessed.[/magenta]")

        console.print(f"[magenta]DEBUG: Checking mime type: {mime}[/magenta]")
        if mime.startswith("video"):
            console.print("[magenta]DEBUG: Mime is video. Accessing video prompts...[/magenta]")
            system_prompt = config["prompts"]["video_system_prompt"]
            prompt = config["prompts"]["video_prompt"]
            console.print("[magenta]DEBUG: Video prompts accessed.[/magenta]")
        elif mime.startswith("audio"):
            console.print("[magenta]DEBUG: Mime is audio. Accessing audio prompts...[/magenta]")
            system_prompt = config["prompts"]["audio_system_prompt"]
            prompt = config["prompts"]["audio_prompt"]
            console.print("[magenta]DEBUG: Audio prompts accessed.[/magenta]")
        elif mime.startswith("image"):
            console.print("[magenta]DEBUG: Mime is image. Accessing image prompts...[/magenta]")
            system_prompt = config["prompts"]["image_system_prompt"]
            prompt = config["prompts"]["image_prompt"]
            console.print("[magenta]DEBUG: Image prompts accessed.[/magenta]")
        else:
            console.print(f"[red]Unsupported mime type for processing: {mime}[/red]")
            console.print("[magenta]DEBUG: EXITING at unsupported mime type[/magenta]")
            return ""
        console.print("[magenta]DEBUG: Prompt selection complete.[/magenta]")

        # === KEEP ONLY GEMINI LOGIC ===
        console.print(f"[magenta]DEBUG: Checking Gemini API key presence (present: {args.gemini_api_key != ''})...[/magenta]")
        if args.gemini_api_key != "":
            console.print("[magenta]DEBUG: Entered Gemini block. Getting generation_config...[/magenta]")
            try:
                generation_config = (
                    config["generation_config"].get(args.gemini_model_path.replace(".", "_"), config["generation_config"]["default"])
                )
                console.print("[magenta]DEBUG: generation_config retrieved.[/magenta]")
            except KeyError as e:
                console.print(f"[red]Error accessing generation config: {e}. Check config.toml.[/red]")
                console.print("[magenta]DEBUG: EXITING at generation config KeyError[/magenta]")
                return ""

            console.print("[magenta]DEBUG: Checking for rate limit...[/magenta]")
            if "rate_limit" in generation_config:
                rate_wait = generation_config.get("rate_wait", 6)
                console.print(f"[yellow]Rate limiting: waiting {rate_wait} seconds...[/yellow]")
                time.sleep(rate_wait)
                console.print("[magenta]DEBUG: Rate limit wait finished.[/magenta]")
            else:
                console.print("[magenta]DEBUG: No rate limit configured.[/magenta]")

            console.print("[magenta]DEBUG: Setting up genai_config...[/magenta]")
            try:
                genai_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=generation_config.get("temperature", 1.0),
                    top_p=generation_config.get("top_p", 0.95),
                    top_k=generation_config.get("top_k", 64),
                    candidate_count=config.get("generation_config", {}).get("candidate_count", 1),
                    max_output_tokens=generation_config.get("max_output_tokens", 8192),
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ],
                    response_mime_type=generation_config.get("response_mime_type", "text/plain"),
                )
                console.print("[magenta]DEBUG: genai_config setup complete.[/magenta]")
            except Exception as config_err:
                 console.print(f"[red]Error setting up Gemini generation config: {config_err}[/red]")
                 console.print("[magenta]DEBUG: EXITING at genai_config setup Exception[/magenta]")
                 return ""

            console.print(f"Using generation_config: {generation_config}")

            console.print("[magenta]DEBUG: Initializing genai.Client...[/magenta]")
            try:
                client = genai.Client(api_key=args.gemini_api_key)
                console.print("[magenta]DEBUG: genai.Client initialized.[/magenta]")
            except Exception as client_err:
                console.print(f"[red]Error initializing Gemini client: {client_err}[/red]")
                console.print("[magenta]DEBUG: EXITING at genai.Client initialization Exception[/magenta]")
                return ""

            # --- Handle File Upload for Large Media ---
            console.print("[cyan]DEBUG: Checking if upload is required...[/cyan]")
            upload_required = (
                mime.startswith("video")
                or mime.startswith("audio")
                and Path(uri).stat().st_size >= 20 * 1024 * 1024
            )
            console.print("[cyan]DEBUG: Finished upload block (or skipped). Proceeding to API call attempts.[/cyan]")
            files = []
            if upload_required:
                console.print(f"[cyan]DEBUG: Checking files for:[/cyan] {uri}")
                try:
                    file = client.files.get(
                        name=sanitize_filename_for_gemini(Path(uri).name)
                    )

                    console.print(file)
                    if file.state.name == "ACTIVE" and (
                        base64.b64decode(file.sha256_hash).decode("utf-8")
                        == sha256hash
                        or file.size_bytes == Path(uri).stat().st_size
                    ):
                        console.print()
                        console.print(
                            f"[green]File {file.name} is already active at {file.uri}[/green]"
                        )
                        files = [file]
                        console.print("[cyan]DEBUG: Files found and uploaded. Proceeding to API call.[/cyan]")
                    else:
                        console.print()
                        console.print(
                            f"[yellow]File {file.name} is already exist but {base64.b64decode(file.sha256_hash).decode('utf-8')} not have same sha256hash {sha256hash}[/yellow]"
                        )
                        client.files.delete(
                            name=sanitize_filename_for_gemini(Path(uri).name)
                        )
                        raise Exception("Delete same name file and retry")

                except Exception as e:
                    console.print(f"[yellow]Failed to get/check file status for {uri}: {e}[/yellow]")
                    console.print(f"[blue]Proceeding to upload file:[/blue] {uri}")
                    files = [upload_to_gemini(client, uri, mime_type=mime)]
                    wait_for_files_active(client, files)
                    console.print("[cyan]DEBUG: Files uploaded. Proceeding to API call.[/cyan]")

            # Some files have a processing delay. Wait for them to be ready.
            # wait_for_files_active(files)
            for attempt in range(args.max_retries):
                try:
                    console.print(f"[blue]Generating captions (Attempt {attempt + 1}/{args.max_retries})...[/blue]")
                    start_time = time.time()

                    # --- Construct API Request Contents ---
                    console.print(f"[cyan]DEBUG: Preparing contents for {uri}...[/cyan]")
                    contents_list = [types.Part.from_text(text=prompt)]

                    # === UNDO TEMP DIAGNOSTIC: Restore original logic START ===
                    if upload_required:
                        if not files: 
                            console.print("[red]ERROR: 'files' list is empty but upload was required.[/red]")
                            raise ValueError("Programming error: Uploaded file reference is missing.")
                        console.print(f"[cyan]DEBUG: Using uploaded file URI: {files[0].uri}[/cyan]")
                        contents_list.insert(0, types.Part.from_uri(file_uri=files[0].uri, mime_type=mime))
                    elif mime.startswith("audio"):
                        console.print("[cyan]DEBUG: Reading small audio file as bytes...[/cyan]")
                        audio_blob = Path(uri).read_bytes()
                        contents_list.insert(0, types.Part.from_bytes(data=audio_blob, mime_type=mime))
                        console.print(f"[cyan]DEBUG: Added audio blob (size: {len(audio_blob)})[/cyan]")
                    elif mime.startswith("image"):
                        console.print("[cyan]DEBUG: Encoding image file as bytes...[/cyan]")
                        blob, pixels = encode_image(uri) 
                        if blob is None:
                            console.print(f"[red]Failed to encode image {uri}. Skipping.[/red]")
                            log_api_call(args.gemini_model_path, "error", time.time() - start_time, "Image encoding failed")
                            return "" 
                        contents_list.insert(0, types.Part.from_bytes(data=blob, mime_type="image/jpeg")) 
                        console.print(f"[cyan]DEBUG: Added image blob (size: {len(blob)})[/cyan]")
                    else:
                        raise ValueError(f"Unsupported mime type for Gemini processing: {mime}")
                    # === UNDO TEMP DIAGNOSTIC: Restore original logic END ===
                    
                    # === REMOVE TEMP DIAGNOSTIC: Forced byte sending START ===
                    # console.print(f"[yellow]DIAGNOSTIC: Attempting to send {mime} as raw bytes...[/yellow]")
                    # try:
                    #     media_blob = Path(uri).read_bytes()
                    #     contents_list.insert(0, types.Part.from_bytes(data=media_blob, mime_type=mime))
                    #     console.print(f"[cyan]DEBUG: Added media blob (size: {len(media_blob)})[/cyan]")
                    # except Exception as read_err:
                    #     console.print(f"[red]DIAGNOSTIC: Failed to read file bytes for {uri}: {read_err}[/red]")
                    #     return "" 
                    # === REMOVE TEMP DIAGNOSTIC: Forced byte sending END ===

                    # --- Make the API Call ---
                    console.print(f"[cyan]DEBUG: Calling generate_content_stream with model {args.gemini_model_path}[/cyan]")
                    console.print(f"[cyan]DEBUG: Contents list length: {len(contents_list)}[/cyan]")
                    response = client.models.generate_content_stream(
                        model=args.gemini_model_path,
                        contents=contents_list,
                        config=genai_config,
                    )
                    console.print(f"[cyan]DEBUG: generate_content_stream call returned. Response object: {type(response)}[/cyan]")

                    # --- Process Streaming Response ---
                    chunks = []
                    console.print("[blue]API Response stream:[/blue]")
                    chunk_received = False # Flag to check if we get any chunks
                    for chunk in response:
                        chunk_received = True # Mark that we received at least one chunk item
                        # TODO: Add handling for potential non-text chunks or errors within the stream if the API supports it
                        if chunk.text:
                            chunks.append(chunk.text)
                            # Print chunk immediately for feedback
                            try:
                                console.print(chunk.text, end="", overflow="ellipsis", highlight=False)
                            except Exception:
                                print(chunk.text, end="")
                            finally:
                                console.file.flush()
                        else:
                            # Log if a chunk doesn't contain text
                            console.print("[yellow]DEBUG: Received non-text chunk part.[/yellow]")

                    if not chunk_received:
                        # Log if the loop finishes without processing any chunks (empty stream?)
                        console.print("[yellow]DEBUG: No chunks received from the response stream.[/yellow]")

                    console.print("\n[green]...stream finished.[/green]") # Indicate end of stream
                    response_text = "".join(chunks).strip() # Combine and strip whitespace

                    # ... (rest of the success path: logging, timing, content assignment) ...
                    elapsed_time = time.time() - start_time
                    log_api_call(args.gemini_model_path, "success", elapsed_time) # Log success

                    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
                    content = response_text
                    if not content:
                         console.print("[yellow]Warning: API returned empty content after processing stream.[/yellow]")
                    return content

                # --- Handle Errors During API Call Attempt ---
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    # Log the exception type as well
                    log_api_call(args.gemini_model_path, "error", elapsed_time, f"{type(e).__name__}: {e}")
                    error_msg = Text(str(e), style="red")
                    # Include traceback for more detailed debugging
                    console.print(f"[red]Error during API call attempt {attempt + 1}/{args.max_retries}: {error_msg}[/red]")
                    console.print_exception(show_locals=True) # Print traceback

                    # ... (rest of error handling: retry logic) ...
                    if attempt < args.max_retries - 1:
                        retry_wait_time = args.wait_time
                        if "429" in str(e) or "resource_exhausted" in str(e).lower() or "rate limit" in str(e).lower():
                             retry_wait_time = max(args.wait_time, 15)
                             console.print(f"[yellow]Rate limit/exhaustion error detected, increasing retry wait to {retry_wait_time}s[/yellow]")
                        time_spent_in_attempt = time.time() - start_time
                        actual_wait = retry_wait_time - time_spent_in_attempt
                        if actual_wait > 0:
                             console.print(f"[yellow]Retrying in {actual_wait:.1f} seconds...[/yellow]")
                             time.sleep(actual_wait)
                        else:
                             console.print("[yellow]Attempt finished quickly after error, retrying immediately...[/yellow]")
                        continue
                    else:
                        console.print(f"[red]Failed to process after {args.max_retries} attempts. Skipping file {uri}.[/red]")
                        return ""
            return ""

        else:
            console.print("[red]Error: Gemini API key not provided.[/red]")
            console.print("[magenta]DEBUG: EXITING because no Gemini key[/magenta]")
            return ""
            
    except Exception as outer_e:
        # Add a top-level exception handler just in case
        console.print("[bold red]***** UNCAUGHT EXCEPTION IN api_process_batch *****[/bold red]")
        console.print_exception(show_locals=True)
        return "" # Return empty on unexpected errors too


def sanitize_filename_for_gemini(name: str) -> str:
    """Sanitizes filenames for Gemini API."""
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized[:40] if len(sanitized) > 40 else sanitized


def upload_to_gemini(client, path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    original_name = Path(path).name
    safe_name = sanitize_filename_for_gemini(original_name)

    file = client.files.upload(
        file=path,
        config=types.UploadFileConfig(
            name=safe_name,
            mime_type=mime_type,
            display_name=original_name,
        ),
    )
    console.print()
    console.print(f"[blue]Uploaded file[/blue] '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(client, files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    console.print("[yellow]Waiting for file processing...[/yellow]")
    for name in (file.name for file in files):
        file = client.files.get(name=name)
        with console.status("[yellow]Processing...[/yellow]", spinner="dots") as status:
            while file.state.name == "PROCESSING":
                time.sleep(10)
                file = client.files.get(name=name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
    console.print("[green]...all files ready[/green]")
    console.print()


def encode_image(image_path: str) -> Optional[Tuple[str, Pixels]]:
    """Encode the image to base64 format with size optimization.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        with Image.open(image_path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]

            # Calculate dimensions that are multiples of 16
            max_size = 1024
            width, height = image.size
            aspect_ratio = width / height

            def calculate_dimensions(max_size: int) -> Tuple[int, int]:
                if width > height:
                    new_width = min(max_size, (width // 16) * 16)
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                else:
                    new_height = min(max_size, (height // 16) * 16)
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                # Ensure dimensions don't exceed max_size
                if new_width > max_size:
                    new_width = max_size
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                if new_height > max_size:
                    new_height = max_size
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                return new_width, new_height

            new_width, new_height = calculate_dimensions(max_size)
            image = image.resize((new_width, new_height), Image.LANCZOS).convert("RGB")

            pixels = Pixels.from_image(
                image,
                resize=(image.width // 18, image.height // 18),
            )

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8"), pixels

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found - {image_path}")
    except Image.UnidentifiedImageError:
        console.print(f"[red]Error:[/red] Cannot identify image file - {image_path}")
    except PermissionError:
        console.print(
            f"[red]Error:[/red] Permission denied accessing file - {image_path}"
        )
    except OSError as e:
        # Specifically handle XMP and metadata-related errors
        if "XMP data is too long" in str(e):
            console.print(
                f"[yellow]Warning:[/yellow] Skipping image with XMP data error - {image_path}"
            )
        else:
            console.print(
                f"[red]Error:[/red] OS error processing file {image_path}: {str(e)}"
            )
    except ValueError as e:
        console.print(
            f"[red]Error:[/red] Invalid value while processing {image_path}: {str(e)}"
        )
    except Exception as e:
        console.print(
            f"[red]Error:[/red] Unexpected error processing {image_path}: {str(e)}"
        )
    return None, None


def process_llm_response(result: str) -> tuple[str, str]:
    """处理LLM返回的结果, 提取短描述和长描述。

    Args:
        result: LLM返回的原始结果文本

    Returns:
        tuple[str, str]: 返回 (short_description, long_description) 元组
    """
    if result and "###" in result:
        short_description, long_description = result.split("###")[-2:]

        # 更彻底地清理描述
        short_description = " ".join(short_description.split(":", 1)[-1].split())
        long_description = " ".join(long_description.split(":", 1)[-1].split())
    else:
        short_description = ""
        long_description = ""

    return short_description, long_description
