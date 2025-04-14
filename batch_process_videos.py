import os
import subprocess
import argparse
import sys
import shutil

# --- Configuration ---
TARGET_FPS = 16
# --- End Configuration ---

"""
Original Dimensions: (1920, 816), Aspect Ratio: 0.4250

--- Pairs within 1.0% tolerance (65 pairs) ---
(112, 48), (224, 96), (304, 128), (336, 144), (416, 176)
(448, 192), (528, 224), (560, 240), (608, 256), (640, 272)
(672, 288), (720, 304), (752, 320), (784, 336), (832, 352)
(864, 368), (896, 384), (912, 384), (944, 400), (976, 416)
(1008, 432), (1024, 432), (1056, 448), (1088, 464), (1120, 480)
(1136, 480), (1168, 496), (1200, 512), (1216, 512), (1232, 528)
(1248, 528), (1280, 544), (1312, 560), (1328, 560), (1344, 576)
(1360, 576), (1392, 592), (1424, 608), (1440, 608), (1456, 624)
(1472, 624), (1504, 640), (1520, 640), (1536, 656), (1552, 656)
(1568, 672), (1584, 672), (1616, 688), (1632, 688), (1648, 704)
(1664, 704), (1680, 720), (1696, 720), (1728, 736), (1744, 736)
(1760, 752), (1776, 752), (1792, 768), (1808, 768), (1824, 768)
(1840, 784), (1856, 784), (1872, 800), (1888, 800), (1904, 816)
"""



# Functions check_ffmpeg_exists and process_video remain the same as in v3
# Paste them here...
def check_ffmpeg_exists(ffmpeg_cmd='ffmpeg'):
    """Checks if the ffmpeg command is accessible in the system's PATH."""
    if shutil.which(ffmpeg_cmd) is None:
        print(f"ERROR: '{ffmpeg_cmd}' command not found in system PATH.", file=sys.stderr)
        print("Please ensure FFmpeg is installed and accessible.", file=sys.stderr)
        return False
    try:
        result = subprocess.run([ffmpeg_cmd, '-version'], check=True, capture_output=True, text=True, timeout=5)
        print(f"Found FFmpeg: {result.stdout.splitlines()[0]}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Warning: '{ffmpeg_cmd} -version' timed out. Assuming ffmpeg exists but is slow/stuck?", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Warning: Could not verify ffmpeg version ('{ffmpeg_cmd} -version' failed: {e}).", file=sys.stderr)
        print("Attempting to proceed anyway.", file=sys.stderr)
        return True

def process_video(input_path, output_path, target_width, target_height, overwrite):
    """
    Processes a single video file using FFmpeg to change FPS and resize.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.
        target_width (int): The desired output width (must be divisible by 16).
        target_height (int): The desired output height (must be divisible by 16).
        overwrite (bool): Whether to overwrite existing output files.

    Returns:
        str: Status of the operation: 'processed', 'skipped', or 'failed'.
    """
    # --- Input File Validation ---
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 'failed'

    # --- Check for existing output file ---
    if not overwrite and os.path.exists(output_path):
        print(f"Skipping: Output file already exists: {os.path.basename(output_path)}")
        return 'skipped'

    # --- Construct FFmpeg Command ---
    command = [
        'ffmpeg',
        '-hide_banner',
        '-i', input_path,
        '-vf', f'fps={TARGET_FPS},scale={target_width}:{target_height}',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-c:a', 'copy',
    ]

    if overwrite:
        command.append('-y')
    else:
        command.append('-n')

    command.extend(['-loglevel', 'warning'])
    command.append(output_path)

    print(f"----------------------------------------------------")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"Outputting to: {os.path.relpath(output_path)}") # Show relative path for clarity
    # print(f"Executing command: {' '.join(command)}")

    # --- Execute FFmpeg ---
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully processed: {os.path.basename(output_path)}")
        return 'processed'
    except subprocess.CalledProcessError as e:
        print(f"Error processing {os.path.basename(input_path)}.", file=sys.stderr)
        print(f"FFmpeg failed with exit code {e.returncode}.", file=sys.stderr)
        print("FFmpeg stderr output:", file=sys.stderr)
        stderr_lines = e.stderr.splitlines()
        max_lines = 20
        if len(stderr_lines) > max_lines:
             print("\n".join(stderr_lines[-max_lines:]), file=sys.stderr)
             print(f"... (truncated {len(stderr_lines) - max_lines} lines)", file=sys.stderr)
        else:
             print(e.stderr, file=sys.stderr)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Removed potentially corrupt output file: {os.path.basename(output_path)}")
            except OSError as rm_err:
                print(f"Warning: Could not remove output file after error: {rm_err}", file=sys.stderr)
        return 'failed'
    except Exception as e:
        print(f"An unexpected Python error occurred processing {os.path.basename(input_path)}: {e}", file=sys.stderr)
        return 'failed'


def main():
    parser = argparse.ArgumentParser(
        description=f"Batch process MP4 videos: change FPS to {TARGET_FPS}, resize, "
                    f"and save results to a subdirectory within the input directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Use default='.' for input_dir, allowing running from CWD without -i
    parser.add_argument("-i", "--input-dir", default='.',
                        help="Directory containing input MP4 files. Defaults to current directory.")
    parser.add_argument("-W", "--width", required=True, type=int,
                        help="Target width for resizing (MUST be multiple of 16).")
    parser.add_argument("-H", "--height", required=True, type=int,
                        help="Target height for resizing (MUST be multiple of 16).")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing files in the output subdirectory.")
    # Removed --output-dir

    args = parser.parse_args()

    # --- Initial Checks ---
    if not check_ffmpeg_exists():
        sys.exit(1)

    if args.width <= 0 or args.height <= 0 or args.width % 16 != 0 or args.height % 16 != 0:
        print(f"Error: Target width ({args.width}) and height ({args.height}) MUST be positive and divisible by 16.", file=sys.stderr)
        sys.exit(1)

    input_dir_abs = os.path.abspath(args.input_dir) # Get absolute path for clarity
    if not os.path.isdir(input_dir_abs):
        print(f"Error: Input directory not found: {input_dir_abs}", file=sys.stderr)
        sys.exit(1)

    # --- Create Output Subdirectory --- (Modification)
    output_subdir_name = f"processed_{TARGET_FPS}fps_{args.width}x{args.height}"
    output_subdir_path = os.path.join(input_dir_abs, output_subdir_name)

    try:
        os.makedirs(output_subdir_path, exist_ok=True)
        print(f"Ensured output subdirectory exists: {output_subdir_path}")
    except OSError as e:
        print(f"Error creating output subdirectory {output_subdir_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Process Files ---
    print(f"\nStarting video processing...")
    print(f"Input directory:  {input_dir_abs}")
    # Print the automatically determined output directory
    print(f"Output directory: {output_subdir_path}")
    print(f"Target size:      {args.width}x{args.height}, Target FPS: {TARGET_FPS}")
    print(f"Overwrite output: {args.overwrite}")
    print(f"====================================================")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    found_mp4 = False

    try:
        # List files from the absolute input directory path
        all_files = sorted(os.listdir(input_dir_abs))
    except OSError as e:
         print(f"Error listing files in input directory '{input_dir_abs}': {e}", file=sys.stderr)
         sys.exit(1)

    for filename in all_files:
        input_path = os.path.join(input_dir_abs, filename)
        # Check if it's a file first, then check extension.
        if os.path.isfile(input_path) and filename.lower().endswith(".mp4"):
            # Ensure it's not already in a subfolder (like the output one)
            if os.path.dirname(input_path) == input_dir_abs:

                # --- Add filter logic here ---
                base, ext = os.path.splitext(filename)
                # Assume clips have at least 2 hyphens (e.g., PREFIX-START-END...)
                # Adjust threshold if your naming convention differs
                hyphen_threshold = 2
                if base.count('-') < hyphen_threshold:
                    print(f"Skipping potential source file (fewer than {hyphen_threshold} hyphens): {filename}")
                    continue # Skip to the next file in the loop
                # --- End filter logic ---

                found_mp4 = True # Now we know it's an MP4 we intend to process

                # Construct output filename (logic moved down slightly)
                # base, ext already defined above
                output_filename = f"{base}{ext}"
                output_path = os.path.join(output_subdir_path, output_filename)

                status = process_video(input_path, output_path, args.width, args.height, args.overwrite)

                if status == 'processed':
                    processed_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                elif status == 'failed':
                    failed_count += 1

    # --- Summary ---
    print(f"\n====================================================")
    if not found_mp4:
         print("No .mp4 files found in the input directory.")
    else:
         print(f"Processing complete.")
         print(f"Successfully processed: {processed_count} file(s)")
         print(f"Skipped (output existed): {skipped_count} file(s)")
         print(f"Failed to process:        {failed_count} file(s)")
    # Report the final output directory path clearly
    print(f"Output files are in: {output_subdir_path}")
    print(f"====================================================")

if __name__ == "__main__":
    main()