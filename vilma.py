import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import mss
import threading
import os
import time
from datetime import datetime
import atexit
import tkinter as tk
from tkinter import filedialog
import platform
import subprocess
from colorama import init, Fore, Style
import ctypes
import pyautogui
from abc import ABC, abstractmethod
import logging
import json

# Initialize colorama (for colored terminal output)
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vilma.log")
    ]
)

# Create logger
logger = logging.getLogger("ViLMA")

# Add additional imports potentially needed for GPU troubleshooting
import gc  # For manual garbage collection
import os  # For environment variables

# Set PyTorch environment variables that might help with GPU detection/usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check if we should run in CPU-only mode (can be useful for debugging)
CPU_ONLY = os.environ.get("VILMA_CPU_ONLY", "").lower() in ("1", "true", "yes")
if CPU_ONLY:
    logger.info("CPU-only mode enabled via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ModelManager:
    """
    Handles loading, preparation, and inference with vision-language models.
    This class is responsible for all interactions with the AI model.
    """

    def __init__(self):
        """
        Initialize the ModelManager with default settings.
        The model and processor will be loaded separately via load_model().
        """
        # More detailed GPU detection
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.cuda_device_name = torch.cuda.get_device_name(0) if self.cuda_available else "None"

        # Default to CPU for maximum compatibility
        self.use_gpu = False
        self.device = torch.device("cpu")

        # Log detailed GPU information
        logger.info(f"CUDA Available: {self.cuda_available}")
        logger.info(f"CUDA Device Count: {self.cuda_device_count}")
        logger.info(f"CUDA Device Name: {self.cuda_device_name}")
        logger.info(f"Default device: {self.device} (use toggle_gpu() to change)")

        # Model and processor references (initialized to None until load_model())
        self.model = None
        self.processor = None

    def toggle_gpu(self):
        """
        Toggle between CPU and GPU usage.
        Will only enable GPU if CUDA is available.

        Returns:
            str: Current device being used
        """
        if not self.cuda_available and not self.use_gpu:
            # Provide detailed diagnostics about why CUDA isn't available
            gpu_info = "\nGPU Diagnostics:\n"

            # Check if PyTorch was built with CUDA
            gpu_info += f"- PyTorch CUDA built: {torch.version.cuda is not None}\n"

            # Try to get NVIDIA driver info on Windows
            if platform.system() == "Windows":
                try:
                    import subprocess
                    nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True, check=False)
                    if nvidia_smi.returncode == 0:
                        gpu_info += "- NVIDIA drivers installed but not detected by PyTorch\n"
                    else:
                        gpu_info += "- NVIDIA drivers not found\n"
                except:
                    gpu_info += "- NVIDIA driver check failed\n"

            # Suggest solutions
            gpu_info += "\nPossible solutions:\n"
            gpu_info += "1. Install an NVIDIA GPU\n"
            gpu_info += "2. Install NVIDIA CUDA drivers\n"
            gpu_info += "3. Reinstall PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"

            logger.warning(f"Cannot enable GPU: CUDA is not available\n{gpu_info}")
            return "cpu (CUDA unavailable)"

        # Toggle the setting
        self.use_gpu = not self.use_gpu if self.cuda_available else False

        # Update device based on setting
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # If model is loaded, move it to the new device
        if self.model is not None:
            try:
                logger.info(f"Moving model to {self.device}")
                self.model = self.model.to(self.device)
                # If using GPU, convert to half precision
                if self.use_gpu:
                    self.model = self.model.half()
                logger.info(f"Model now on: {next(self.model.parameters()).device}")
            except Exception as e:
                logger.error(f"Error moving model to {self.device}: {e}")

        return str(self.device)

    def load_model(self, model_path):
        """
        Loads the model and processor from a given model directory.

        Args:
            model_path (str): Path to the HuggingFace model directory on disk.

        Returns:
            bool: True if model loaded successfully, False otherwise.

        Raises:
            RuntimeError: If there's an error during model loading.
        """
        try:
            logger.info(f"Loading model from: {model_path}")

            # If using CUDA, make sure cache is clear
            if self.cuda_available:
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory before loading: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

            # Load the Causal LM model in eval mode
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            ).eval()

            # Explicitly move model to device
            self.model = self.model.to(self.device)

            # Double-check model device placement
            logger.info(f"Model loaded to: {next(self.model.parameters()).device}")

            # If we have a GPU available, convert the model to half-precision for speed
            if self.cuda_available:
                logger.info("Converting model to half precision (FP16)")
                self.model = self.model.half()
                logger.info(f"CUDA memory after loading: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

            # Load the corresponding processor (tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")

    def prepare_inputs(self, task_prompt, image):
        """
        Prepares the text and image for the model using the processor.

        Args:
            task_prompt (str): Text prompt to guide the model
            image (PIL.Image): Screenshot or window capture to analyze

        Returns:
            dict: Tensors ready for inference

        Raises:
            RuntimeError: If there's an error during input preparation
        """
        try:
            # Processor will handle tokenization and any image transformation
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")

            # Explicitly move to the correct device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
                    logger.debug(f"Input '{k}' moved to device: {v.device}")

            # On CUDA, optionally cast float tensors to half precision
            if self.use_gpu and self.cuda_available:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                        inputs[k] = v.half()

            return inputs
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            raise RuntimeError(f"Error preparing inputs: {e}")

    def run_model(self, inputs):
        """
        Run the model forward pass (generate) on the prepared inputs.

        Args:
            inputs: Output of prepare_inputs()

        Returns:
            torch.Tensor: Generated token ids from the model

        Raises:
            RuntimeError: If there's an error during model inference
        """
        try:
            # Log memory usage before inference if using CUDA
            if self.use_gpu and self.cuda_available:
                logger.debug(f"CUDA memory before inference: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                logger.debug(f"CUDA memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

            # Ensure model is in eval mode
            self.model.eval()

            # Check device semantically, not just by string comparison
            model_device = next(self.model.parameters()).device

            # Double-check all inputs are on correct device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    # Only move if actually on different device (ignoring string differences like 'cuda' vs 'cuda:0')
                    if v.device.type != model_device.type or (
                       v.device.type == 'cuda' and model_device.type == 'cuda' and
                       v.device.index != model_device.index and model_device.index is not None):
                        logger.info(f"Moving input {k} from {v.device} to {model_device}")
                        inputs[k] = v.to(model_device)

            # Use automatic mixed precision on GPU
            with torch.inference_mode(), torch.amp.autocast(
                device_type=model_device.type,
                enabled=self.use_gpu and self.cuda_available
            ):
                # Time the inference
                start_time = time.time()

                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )

                end_time = time.time()

            # Log performance metrics
            inference_time = end_time - start_time
            logger.info(f"Inference took {inference_time:.2f} seconds")

            if self.use_gpu and self.cuda_available:
                logger.debug(f"CUDA memory after inference: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

            return generated_ids
        except Exception as e:
            logger.error(f"Error running model: {e}")
            raise RuntimeError(f"Error running model: {e}")

    def process_outputs(self, generated_ids):
        """
        Decodes the model's output token IDs back into text.

        Args:
            generated_ids: The output from run_model()

        Returns:
            str: Decoded text from the model

        Raises:
            RuntimeError: If there's an error during output processing
        """
        try:
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            return generated_text
        except Exception as e:
            logger.error(f"Error processing outputs: {e}")
            raise RuntimeError(f"Error processing outputs: {e}")

    def run_inference(self, image, prompt):
        """
        Full pipeline of: prepare -> run -> decode on a single image+prompt pair.

        Args:
            image (PIL.Image): The image to analyze
            prompt (str): The text prompt to guide the model

        Returns:
            str: The cleaned model output or 'Error' if something fails
        """
        try:
            # Check if model is loaded
            if self.model is None or self.processor is None:
                logger.error("Model not loaded. Please load model before inference.")
                return "Error"

            # Prepare inputs from the user-defined prompt plus the screenshot
            inputs = self.prepare_inputs(prompt, image)
            # Generate token IDs from the model
            generated_ids = self.run_model(inputs)
            # Decode back into text
            generated_text = self.process_outputs(generated_ids)

            # Clean up known artifacts or tokens
            cleaned_text = generated_text.replace("</s><s>", "").strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Raw Inference result: {generated_text}")
            logger.info(f"Cleaned Inference result: {cleaned_text}")

            return cleaned_text
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return "Error"


class ScreenCapture:
    """
    Handles all screen and window capturing functionality, including
    screenshot taking and video recording.
    """

    def __init__(self):
        """
        Initialize the ScreenCapture module with default settings.
        """
        # If set, only capture from a single window with a known title
        self.target_window = None

        # Resolution setting for image processing/resizing: "640p", "720p", "1080p", "native"
        self.resolution = "720p"

        # Recording variables
        self._recording_event = threading.Event()
        self.current_video_filename = None
        self.video_writer = None

        # Set the recording event initially to indicate we're not recording
        self._recording_event.set()

    @property
    def is_recording(self):
        """
        Property that returns True if the recording thread is active.
        We interpret "active" as _recording_event NOT being set.
        - Event is "cleared" => record_desktop() loops.
        - Event is "set" => record_desktop() thread is signaled to stop.

        Returns:
            bool: True if recording is active, False otherwise
        """
        return not self._recording_event.is_set()

    def capture_desktop(self):
        """
        Captures either the full desktop or a specific window, based on self.target_window.
        Uses mss to grab the screen contents.

        Returns:
            PIL.Image: The captured screenshot

        Raises:
            RuntimeError: If there's an error during screen capture
        """
        try:
            with mss.mss() as sct:
                if self.target_window:
                    # If a target window is specified, find a window matching that title via PyAutoGUI
                    window = pyautogui.getWindowsWithTitle(self.target_window)
                    if window:
                        window = window[0]
                        left, top, width, height = window.left, window.top, window.width, window.height
                        monitor = {"top": top, "left": left, "width": width, "height": height}
                    else:
                        logger.warning(f"Window '{self.target_window}' not found. Capturing full desktop.")
                        monitor = sct.monitors[1]
                else:
                    # Otherwise capture the primary monitor
                    monitor = sct.monitors[1]

                screenshot = sct.grab(monitor)
                # Convert raw BGRA data from MSS into a PIL image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
        except Exception as e:
            logger.error(f"Error capturing desktop: {e}")
            raise RuntimeError(f"Error capturing desktop: {e}")

    def take_screenshot(self):
        """
        Takes a screenshot and saves it to disk with a timestamped filename.

        Returns:
            str: Path to the saved screenshot or None if failed
        """
        try:
            screenshot = self.capture_desktop()
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot.save(filename)
            logger.info(f"Screenshot taken: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None

    def start_recording(self):
        """
        Starts recording the screen (desktop or window) in a separate thread.

        Returns:
            str: Path to the recording file or None if failed
        """
        try:
            # Ensure the event is cleared so the recording thread runs
            self._recording_event.clear()

            # Prepare VideoWriter with mp4v codec at ~20 FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width, height = self.get_resolution_dimensions()
            self.current_video_filename = f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'

            self.video_writer = cv2.VideoWriter(self.current_video_filename, fourcc, 20.0, (width, height))

            logger.info(f"Recording started. Output file: {self.current_video_filename}")
            logger.info(f"Stored in directory: {os.getcwd()}")

            # Launch the dedicated recording thread
            record_thread = threading.Thread(target=self.record_desktop, daemon=True)
            record_thread.start()

            return self.current_video_filename
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return None

    def record_desktop(self):
        """
        Continuously captures frames while _recording_event is NOT set.
        Writes frames to the self.video_writer.
        """
        try:
            with mss.mss() as sct:
                while not self._recording_event.is_set():
                    try:
                        # Attempt to capture from desktop or specific window
                        screenshot = self.capture_desktop()
                    except Exception as capture_err:
                        logger.error(f"Error capturing desktop while recording: {capture_err}")
                        continue

                    # Convert from PIL to numpy (RGB -> BGR for OpenCV)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Write frames out if the writer is still open
                    if self.video_writer:
                        self.video_writer.write(frame)

                    # Limit framerate to ~20 FPS
                    time.sleep(1/20)
        except Exception as e:
            logger.error(f"Error recording desktop: {e}")
        finally:
            # Ensure release of the video writer resource
            with self._video_writer_lock():
                pass
            logger.info("Recording thread has exited.")

    def _video_writer_lock(self):
        """
        Context manager for safely releasing video writer resources.
        """
        class VideoWriterLock:
            def __init__(self, outer):
                self.outer = outer

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.outer.video_writer:
                    self.outer.video_writer.release()
                    self.outer.video_writer = None

        return VideoWriterLock(self)

    def stop_recording(self):
        """
        Signals the recording thread to stop if it is currently running.

        Returns:
            str: Path to the completed recording file or None if not recording
        """
        try:
            if self.is_recording:
                self._recording_event.set()
                logger.info(f"Recording stop requested. Final file: {self.current_video_filename}")
                return self.current_video_filename
            return None
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None

    def get_resolution_dimensions(self):
        """
        Returns (width, height) based on the selected resolution setting
        or the target window dimensions if applicable.

        Returns:
            tuple: (width, height) in pixels
        """
        # If a specific window is targeted, try to get that window's geometry
        if self.target_window:
            window = pyautogui.getWindowsWithTitle(self.target_window)
            if window:
                window = window[0]
                return window.width, window.height

        # Otherwise, pick a resolution from our set of known strings
        if self.resolution == "640p":
            return 640, 360
        elif self.resolution == "720p":
            return 1280, 720
        elif self.resolution == "1080p":
            return 1920, 1080
        elif self.resolution == "native":
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                return monitor["width"], monitor["height"]
        else:
            # Default fallback = 720p
            return 1280, 720

    def set_target_window(self, window_title):
        """
        Sets the target window to capture from.

        Args:
            window_title (str): Title of the window to capture or None for full desktop

        Returns:
            bool: True if window was found (or None was passed), False otherwise
        """
        if window_title is None:
            self.target_window = None
            logger.info("Target set to full desktop")
            return True

        # Verify window exists
        windows = pyautogui.getWindowsWithTitle(window_title)
        if windows:
            self.target_window = window_title
            logger.info(f"Target window set to: {window_title}")
            return True
        else:
            logger.warning(f"Window '{window_title}' not found")
            return False

    def toggle_resolution(self):
        """
        Cycles through a list of supported resolutions.

        Returns:
            str: The new resolution setting
        """
        resolutions = ["640p", "720p", "1080p", "native"]
        current_index = resolutions.index(self.resolution)
        new_index = (current_index + 1) % len(resolutions)
        self.resolution = resolutions[new_index]
        logger.info(f"Resolution set to {self.resolution}")
        return self.resolution


class ActionHandler:
    """
    Handles all actions that can be triggered based on inference results,
    such as logging out, showing blank screens, keyboard commands, etc.
    """

    def __init__(self, screen_capture):
        """
        Initialize the ActionHandler with references to other components.

        Args:
            screen_capture (ScreenCapture): Reference to the screen capture module
        """
        self.screen_capture = screen_capture

        # Flags for different actions
        self.logout_on_trigger = False
        self.blank_screen_on_trigger = False
        self.screenshot_on_trigger = False
        self.record_on_trigger = False
        self.dummy_mode = False

        # For custom trigger logic (open a file/app when a certain text is detected)
        self.custom_trigger_path = None
        self.custom_trigger_enabled = False
        self.custom_trigger_output = "yes"

        # Keyboard trigger logic (type a sequence if a certain text is detected)
        self.keyboard_trigger_enabled = False
        self.keyboard_trigger_sequence = ""
        self.keyboard_trigger_activated = False
        self.keyboard_trigger_output = "yes"

        # Blank window state
        self.blank_window_open = False

        # Ensure we close the blank screen if the program exits unexpectedly
        atexit.register(self.ensure_blank_window_closed)

    def process_inference_result(self, result):
        """
        Process the inference result and trigger appropriate actions.

        Args:
            result (str): The inference result text

        Returns:
            bool: True if any action was triggered, False otherwise
        """
        any_action_triggered = False

        # If we get "yes" and not in dummy mode
        if result.lower() == "yes" and not self.dummy_mode:
            # Optionally log out
            if self.logout_on_trigger:
                logger.info("Trigger detected, logging out")
                self.logout()
                any_action_triggered = True

            # Optionally show blank window
            if self.blank_screen_on_trigger and not self.blank_window_open:
                logger.info("Trigger detected, opening blank window")
                # Start blank window in a separate thread to avoid blocking
                blank_thread = threading.Thread(target=self.show_blank_window, daemon=True)
                blank_thread.start()
                any_action_triggered = True

            # Optionally take a screenshot
            if self.screenshot_on_trigger:
                self.screen_capture.take_screenshot()
                any_action_triggered = True

            # If record_on_trigger is on, start recording if not already
            if self.record_on_trigger and not self.screen_capture.is_recording:
                self.screen_capture.start_recording()
                any_action_triggered = True

        # Check if the custom trigger is enabled and matches
        if self.custom_trigger_enabled and self.custom_trigger_path:
            if self.custom_trigger_output.lower() in result.lower():
                logger.info("Custom trigger matched. Running custom trigger.")
                self.run_custom_trigger()
                # After the first match, disable custom trigger automatically
                self.custom_trigger_enabled = False
                any_action_triggered = True

        # Check if the "keyboard trigger" is enabled and matches
        if self.keyboard_trigger_enabled and (self.keyboard_trigger_output.lower() in result.lower()) and not self.keyboard_trigger_activated:
            logger.info("Keyboard trigger matched. Running keyboard trigger.")
            self.run_keyboard_trigger()
            self.keyboard_trigger_activated = True
            any_action_triggered = True

        # If we get "no", we close the blank window and stop recording if they are active
        elif result.lower() == "no":
            if self.blank_window_open:
                logger.info("No trigger detected, closing blank window")
                self.ensure_blank_window_closed()
                any_action_triggered = True

            if self.record_on_trigger and self.screen_capture.is_recording:
                self.screen_capture.stop_recording()
                any_action_triggered = True

        # If the text no longer contains the keyboard trigger target, reset the activation flag
        if self.keyboard_trigger_enabled and self.keyboard_trigger_output.lower() not in result.lower():
            self.keyboard_trigger_activated = False

        return any_action_triggered

    def run_custom_trigger(self):
        """
        Execute a custom trigger action to open a file or application.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Running custom trigger: {self.custom_trigger_path}")

            # Use different commands depending on Windows/Darwin/Linux
            if platform.system() == "Windows":
                os.startfile(self.custom_trigger_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", self.custom_trigger_path], check=True)
            else:
                subprocess.run(["xdg-open", self.custom_trigger_path], check=True)

            logger.info("Custom trigger executed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error running custom trigger: {e}")
            return False

    def run_keyboard_trigger(self):
        """
        Execute a keyboard sequence using PyAutoGUI.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            pyautogui.typewrite(self.keyboard_trigger_sequence)
            logger.info(f"Executed keyboard sequence: {self.keyboard_trigger_sequence}")
            return True
        except Exception as e:
            logger.error(f"Error executing keyboard trigger: {e}")
            return False

    def show_blank_window(self):
        """
        Creates a fullscreen black window to obscure the screen.

        Returns:
            bool: True if window was shown, False on error
        """
        try:
            logger.info("Showing blank window")

            self.blank_window_open = True

            # Create an all-black image
            blank_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Create OpenCV window with context manager for safety
            cv2.namedWindow("Blank Screen", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Blank Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # On Windows, force the window to front/topmost
            if platform.system() == "Windows":
                hwnd = ctypes.windll.user32.FindWindowW(None, "Blank Screen")
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

            # Continue showing the window until user presses 'q' or blank_window_open becomes False
            while self.blank_window_open:
                cv2.imshow("Blank Screen", blank_screen)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.blank_window_open = False
                    break

            # Once done, destroy the window
            cv2.destroyWindow("Blank Screen")
            logger.info("Closed blank window")
            return True
        except Exception as e:
            logger.error(f"Error showing blank window: {e}")
            return False

    def ensure_blank_window_closed(self):
        """
        Safely close the blank window if it's still open.
        Called on program exit or when clearing the screen.
        """
        try:
            self.blank_window_open = False
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error closing blank window: {e}")

    def logout(self):
        """
        Logs the user out of the system on Windows, macOS, or Linux.

        Returns:
            bool: True if successful, False on error or unsupported platform
        """
        try:
            system_platform = platform.system()
            if system_platform == "Windows":
                subprocess.run(["shutdown", "/l"], check=True)
            elif system_platform in ("Linux", "Darwin"):
                subprocess.run(["pkill", "-KILL", "-u", os.getlogin()], check=True)
            else:
                logger.warning(f"Unsupported operating system: {system_platform}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error logging out: {e}")
            return False


class ConfigManager:
    """
    Manages configuration settings, including saving/loading
    preferences, inference prompts, and other settings.
    """

    def __init__(self, config_file="vilma_config.json"):
        """
        Initialize the ConfigManager with default settings.

        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.prompts = []
        self.inference_rate = None

        # Try to load existing configuration
        self.load_config()

    def load_config(self):
        """
        Load configuration from the config file if it exists.

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                self.prompts = config.get('prompts', [])
                self.inference_rate = config.get('inference_rate', None)

                logger.info(f"Loaded configuration from {self.config_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def save_config(self):
        """
        Save current configuration to the config file.

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            config = {
                'prompts': self.prompts,
                'inference_rate': self.inference_rate
            }

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def add_prompt(self, prompt):
        """
        Add an inference prompt to the list.

        Args:
            prompt (str): The prompt to add

        Returns:
            bool: True if added successfully, False otherwise
        """
        if prompt and prompt not in self.prompts:
            self.prompts.append(prompt)
            self.save_config()
            return True
        return False

    def remove_prompt(self, index):
        """
        Remove an inference prompt by index.

        Args:
            index (int): Index of the prompt to remove

        Returns:
            str: The removed prompt if successful, None otherwise
        """
        try:
            if 0 <= index < len(self.prompts):
                removed = self.prompts.pop(index)
                self.save_config()
                return removed
            return None
        except Exception as e:
            logger.error(f"Error removing prompt: {e}")
            return None

    def set_inference_rate(self, rate):
        """
        Set the inference rate (frames per second).

        Args:
            rate (int or None): Target inference rate or None for max speed

        Returns:
            bool: True if set successfully, False otherwise
        """
        try:
            if rate is not None and not (1 <= rate <= 30):
                return False

            self.inference_rate = rate
            self.save_config()
            return True
        except Exception as e:
            logger.error(f"Error setting inference rate: {e}")
            return False


class UserInterface:
    """
    Handles all user interaction, including menus, input/output, and displaying status.
    """

    def __init__(self, model_manager, screen_capture, action_handler, config_manager):
        """
        Initialize the UserInterface with references to other components.

        Args:
            model_manager (ModelManager): Reference to the model manager
            screen_capture (ScreenCapture): Reference to the screen capture module
            action_handler (ActionHandler): Reference to the action handler
            config_manager (ConfigManager): Reference to the configuration manager
        """
        self.model_manager = model_manager
        self.screen_capture = screen_capture
        self.action_handler = action_handler
        self.config_manager = config_manager

    def toggle_gpu_menu(self):
        """
        Toggle between CPU and GPU usage.
        """
        try:
            current_device = self.model_manager.toggle_gpu()

            if self.model_manager.use_gpu:
                print(Fore.GREEN + f"GPU usage enabled. Using device: {current_device}" + Style.RESET_ALL)
            else:
                if not self.model_manager.cuda_available:
                    print(Fore.YELLOW + f"GPU usage not available: CUDA not detected" + Style.RESET_ALL)
                    print(Fore.CYAN + "Check the log file for detailed diagnostics and possible solutions." + Style.RESET_ALL)
                    print(Fore.GREEN + "Continuing with CPU mode (which works well for many use cases)." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"GPU usage disabled. Using device: {current_device}" + Style.RESET_ALL)

            # If model is loaded, remind user they might need to reload
            if self.model_manager.model is not None:
                print(Fore.CYAN + "Note: For best results, you may want to reload your model with this new setting." + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"Error toggling GPU usage: {e}" + Style.RESET_ALL)

    def display_banner(self):
        """
        Display the welcome banner.
        """
        print(Fore.CYAN + "\n=== Welcome to ViLMA (Vision-Language Model-based Active Monitoring) ===" + Style.RESET_ALL)
        print(Fore.CYAN + "== A system to monitor screens and perform actions based on what it sees ==" + Style.RESET_ALL)

    def display_menu(self):
        """
        Display the main menu options.
        """
        print(Fore.CYAN + "\n=== Menu ===" + Style.RESET_ALL)

        print(Fore.LIGHTGREEN_EX + "1. Start Screen Monitoring" + Style.RESET_ALL)

        print(Fore.MAGENTA + "\nModel Operations:" + Style.RESET_ALL)
        print(Fore.LIGHTMAGENTA_EX + "2. Load Model" + Style.RESET_ALL)
        print(Fore.LIGHTMAGENTA_EX + "3. Toggle GPU Usage (current: " +
              (Fore.GREEN + "ON" if self.model_manager.use_gpu else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)

        print(Fore.BLUE + "\nMonitoring Settings:" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "4. Add Inference Prompt" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "5. Remove Inference Prompt" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "6. List Inference Prompts" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "7. Set Inference Rate (current: " +
              (Fore.GREEN + str(self.config_manager.inference_rate) if self.config_manager.inference_rate else Fore.RED + "None") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "8. Toggle Processing Resolution (current: " +
              Fore.GREEN + self.screen_capture.resolution + Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "9. Set Target Window (current: " +
              (Fore.GREEN + self.screen_capture.target_window if self.screen_capture.target_window else Fore.RED + "Full Desktop") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)

        print(Fore.GREEN + "\nToggles & Triggers:" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "10. Logout (current: " +
              (Fore.GREEN + "ON" if self.action_handler.logout_on_trigger else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "11. Dummy Mode (current: " +
              (Fore.GREEN + "ON" if self.action_handler.dummy_mode else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "12. Blank Screen (current: " +
              (Fore.GREEN + "ON" if self.action_handler.blank_screen_on_trigger else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "13. Screenshot (current: " +
              (Fore.GREEN + "ON" if self.action_handler.screenshot_on_trigger else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "14. Record (current: " +
              (Fore.GREEN + "ON" if self.action_handler.record_on_trigger else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "15. Custom (current: " +
              (Fore.GREEN + "ON" if self.action_handler.custom_trigger_enabled else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + "16. Keyboard Command (current: " +
              (Fore.GREEN + "ON" if self.action_handler.keyboard_trigger_enabled else Fore.RED + "OFF") +
              Style.RESET_ALL + ")" + Style.RESET_ALL)

        print(Fore.YELLOW + "\nGeneral:" + Style.RESET_ALL)
        print(Fore.LIGHTYELLOW_EX + "17. Quit" + Style.RESET_ALL)

        print(Fore.CYAN + "\n==========================" + Style.RESET_ALL)

    def terminal_menu(self):
        """
        Display the main menu and handle user input in a loop.
        """
        self.display_banner()

        while True:
            self.display_menu()
            choice = input("Enter your choice: ")

            try:
                if choice == "1":
                    self.start_monitoring()

                elif choice == "2":
                    self.load_model_menu()

                elif choice == "3":
                    self.toggle_gpu_menu()

                elif choice == "4":
                    self.add_prompt_menu()

                elif choice == "5":
                    self.remove_prompt_menu()

                elif choice == "6":
                    self.list_prompts()

                elif choice == "7":
                    self.set_inference_rate_menu()

                elif choice == "8":
                    self.toggle_resolution_menu()

                elif choice == "9":
                    self.set_target_window_menu()

                elif choice == "10":
                    self.toggle_logout()

                elif choice == "11":
                    self.toggle_dummy_mode()

                elif choice == "12":
                    self.toggle_blank_screen()

                elif choice == "13":
                    self.toggle_screenshot()

                elif choice == "14":
                    self.toggle_record()

                elif choice == "15":
                    self.configure_custom_trigger()

                elif choice == "16":
                    self.configure_keyboard_trigger()

                elif choice == "17":
                    print(Fore.CYAN + "Quitting..." + Style.RESET_ALL)
                    # Ensure all resources are released
                    self.action_handler.ensure_blank_window_closed()
                    if self.screen_capture.is_recording:
                        self.screen_capture.stop_recording()
                    break

                else:
                    print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)

            except Exception as e:
                logger.error(f"Error in menu option {choice}: {e}")
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

    def start_monitoring(self):
        """
        Start the main monitoring loop if all prerequisites are met.
        """
        # Check if model is loaded
        if self.model_manager.model is None:
            print(Fore.RED + "Error: No model loaded. Please load a model before starting monitoring." + Style.RESET_ALL)
            return

        # Check if prompts are set
        if not self.config_manager.prompts:
            print(Fore.RED + "Error: No inference prompts set. Please add at least one inference prompt before starting monitoring." + Style.RESET_ALL)
            return

        print(Fore.CYAN + "Starting screen monitoring..." + Style.RESET_ALL)
        print(Fore.CYAN + "Press 'q' in any OpenCV window to stop monitoring." + Style.RESET_ALL)

        try:
            while True:
                start_time = time.time()
                try:
                    # Capture the screen
                    screen = self.screen_capture.capture_desktop()
                except RuntimeError as cap_err:
                    # If capturing fails, wait briefly and retry
                    logger.error(f"Error capturing desktop: {cap_err}")
                    time.sleep(1)
                    continue

                # Convert the screenshot to RGB -> numpy -> resized -> back to PIL
                screen_rgb = screen.convert("RGB")
                screen_np = np.array(screen_rgb)
                width, height = self.screen_capture.get_resolution_dimensions()
                screen_resized = cv2.resize(screen_np, (width, height))
                pil_image = Image.fromarray(screen_resized)

                # Run each prompt in the prompts list
                for prompt in self.config_manager.prompts:
                    result = self.model_manager.run_inference(pil_image, prompt)

                    # Process the result and trigger actions if needed
                    self.action_handler.process_inference_result(result)

                # Respect the inference_rate setting if it's not None
                elapsed_time = time.time() - start_time
                if self.config_manager.inference_rate:
                    time_to_wait = max(1.0 / self.config_manager.inference_rate - elapsed_time, 0)
                    time.sleep(time_to_wait)

                # Check if user pressed 'q' in any open CV window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
        finally:
            # Make sure we clean up resources
            self.action_handler.ensure_blank_window_closed()
            if self.screen_capture.is_recording:
                self.screen_capture.stop_recording()

    def load_model_menu(self):
        """
        Open a file dialog to select a model directory and load the model.
        """
        print(Fore.CYAN + "Select the model directory containing the model files..." + Style.RESET_ALL)
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askdirectory(title="Select Model Directory")
        root.destroy()

        if not model_path:
            print(Fore.RED + "Model path selection cancelled." + Style.RESET_ALL)
            return

        try:
            self.model_manager.load_model(model_path)
            print(Fore.GREEN + "Model loaded successfully." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error loading model: {e}" + Style.RESET_ALL)

    def add_prompt_menu(self):
        """
        Add a new inference prompt to the configuration.
        """
        prompt = input("Enter the inference prompt to add: ")
        if prompt:
            if self.config_manager.add_prompt(prompt):
                print(Fore.GREEN + f"Added inference prompt: {prompt}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + "Prompt already exists or is empty." + Style.RESET_ALL)
        else:
            print(Fore.RED + "Cannot add empty prompt." + Style.RESET_ALL)

    def remove_prompt_menu(self):
        """
        Remove an existing inference prompt from the configuration.
        """
        if not self.config_manager.prompts:
            print(Fore.YELLOW + "No prompts to remove." + Style.RESET_ALL)
            return

        self.list_prompts()
        try:
            index = int(input("Enter the prompt number to remove: ")) - 1
            removed = self.config_manager.remove_prompt(index)
            if removed:
                print(Fore.GREEN + f"Removed inference prompt: {removed}" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Invalid prompt number." + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "Please enter a valid number." + Style.RESET_ALL)

    def list_prompts(self):
        """
        Display all configured inference prompts.
        """
        print(Fore.CYAN + "\nCurrent inference prompts:" + Style.RESET_ALL)
        if not self.config_manager.prompts:
            print(Fore.YELLOW + "No prompts configured." + Style.RESET_ALL)
            return

        for i, prompt in enumerate(self.config_manager.prompts, 1):
            print(Fore.GREEN + f"{i}. {prompt}" + Style.RESET_ALL)

    def set_inference_rate_menu(self):
        """
        Set the inference rate (frames per second).
        """
        try:
            rate_input = input("Enter the desired inference rate (1-30, or 'None' for max speed): ")

            if rate_input.lower() == 'none':
                rate = None
            else:
                rate = int(rate_input)
                if not (1 <= rate <= 30):
                    print(Fore.RED + "Invalid rate. Please enter a number between 1 and 30." + Style.RESET_ALL)
                    return

            if self.config_manager.set_inference_rate(rate):
                print(Fore.GREEN + f"Inference rate set to {rate if rate is not None else 'max speed'}." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Failed to set inference rate." + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "Please enter a valid number or 'None'." + Style.RESET_ALL)

    def toggle_resolution_menu(self):
        """
        Cycle through the available resolution options.
        """
        new_resolution = self.screen_capture.toggle_resolution()
        print(Fore.GREEN + f"Resolution set to {new_resolution}." + Style.RESET_ALL)

    def set_target_window_menu(self):
        """
        Set the target window to capture from.
        """
        all_windows = pyautogui.getAllWindows()

        # Filter out windows with empty or whitespace-only titles
        titled_windows = []
        for idx, w in enumerate(all_windows):
            title = w.title.strip()
            if title:
                titled_windows.append((idx, title))

        if not titled_windows:
            print(Fore.YELLOW + "No titled windows found. Defaulting to full desktop capture." + Style.RESET_ALL)
            self.screen_capture.set_target_window(None)
            return

        print(Fore.CYAN + "Open Windows:" + Style.RESET_ALL)
        for local_index, (original_idx, title) in enumerate(titled_windows):
            print(f"[{local_index}] {title}")

        selection = input("\nEnter the index of the window you want to monitor (leave blank for full desktop): ")

        if selection.strip() == "":
            print(Fore.GREEN + "No selection, capturing full desktop." + Style.RESET_ALL)
            self.screen_capture.set_target_window(None)
            return

        try:
            selection_index = int(selection)
            if 0 <= selection_index < len(titled_windows):
                chosen_title = titled_windows[selection_index][1]
                if self.screen_capture.set_target_window(chosen_title):
                    print(Fore.GREEN + f"Target window set to: {chosen_title}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"Failed to set target window: {chosen_title}" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Invalid index. Capturing full desktop." + Style.RESET_ALL)
                self.screen_capture.set_target_window(None)
        except ValueError:
            print(Fore.RED + "Invalid input. Defaulting to full desktop." + Style.RESET_ALL)
            self.screen_capture.set_target_window(None)

    def toggle_logout(self):
        """
        Toggle the logout on trigger feature.
        """
        self.action_handler.logout_on_trigger = not self.action_handler.logout_on_trigger
        print(Fore.GREEN + f"Logout on Trigger is now {'ON' if self.action_handler.logout_on_trigger else 'OFF'}" + Style.RESET_ALL)

    def toggle_dummy_mode(self):
        """
        Toggle dummy mode (no real actions).
        """
        self.action_handler.dummy_mode = not self.action_handler.dummy_mode
        print(Fore.GREEN + f"Dummy mode is now {'ON' if self.action_handler.dummy_mode else 'OFF'}" + Style.RESET_ALL)

    def toggle_blank_screen(self):
        """
        Toggle the blank screen on trigger feature.
        """
        self.action_handler.blank_screen_on_trigger = not self.action_handler.blank_screen_on_trigger
        print(Fore.GREEN + f"Blank Screen on Trigger is now {'ON' if self.action_handler.blank_screen_on_trigger else 'OFF'}" + Style.RESET_ALL)

    def toggle_screenshot(self):
        """
        Toggle the screenshot on trigger feature.
        """
        self.action_handler.screenshot_on_trigger = not self.action_handler.screenshot_on_trigger
        print(Fore.GREEN + f"Screenshot on Trigger is now {'ON' if self.action_handler.screenshot_on_trigger else 'OFF'}" + Style.RESET_ALL)

    def toggle_record(self):
        """
        Toggle the record on trigger feature.
        """
        self.action_handler.record_on_trigger = not self.action_handler.record_on_trigger
        print(Fore.GREEN + f"Record on Trigger is now {'ON' if self.action_handler.record_on_trigger else 'OFF'}" + Style.RESET_ALL)

    def configure_custom_trigger(self):
        """
        Configure a custom file/application trigger.
        """
        if not self.action_handler.custom_trigger_enabled:
            root = tk.Tk()
            root.withdraw()
            custom_trigger_path = filedialog.askopenfilename(title="Select File to Open on Trigger")
            root.destroy()

            if custom_trigger_path:
                custom_trigger_output = input("Enter the output that triggers the custom action (e.g., 'yes', 'no', 'open file', etc.): ")
                self.action_handler.custom_trigger_path = custom_trigger_path
                self.action_handler.custom_trigger_output = custom_trigger_output
                self.action_handler.custom_trigger_enabled = True
                print(Fore.GREEN + f"Custom Trigger set to open {custom_trigger_path} on output: {custom_trigger_output}" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Custom Trigger path selection cancelled." + Style.RESET_ALL)
        else:
            self.action_handler.custom_trigger_enabled = False
            self.action_handler.custom_trigger_path = None
            self.action_handler.custom_trigger_output = "yes"
            print(Fore.GREEN + "Custom Trigger is now OFF" + Style.RESET_ALL)

    def configure_keyboard_trigger(self):
        """
        Configure a keyboard trigger action.
        """
        if not self.action_handler.keyboard_trigger_enabled:
            keyboard_sequence = input("Enter the keyboard sequence to type on trigger: ")
            keyboard_output = input("Enter the output that triggers the keyboard action (e.g., 'yes', 'no', 'low health', etc.): ")

            self.action_handler.keyboard_trigger_sequence = keyboard_sequence
            self.action_handler.keyboard_trigger_output = keyboard_output
            self.action_handler.keyboard_trigger_enabled = True
            self.action_handler.keyboard_trigger_activated = False

            print(Fore.GREEN + f"Keyboard Trigger set to type: {keyboard_sequence} on output: {keyboard_output}" + Style.RESET_ALL)
        else:
            self.action_handler.keyboard_trigger_enabled = False
            self.action_handler.keyboard_trigger_sequence = ""
            self.action_handler.keyboard_trigger_output = "yes"
            print(Fore.GREEN + "Keyboard Trigger is now OFF" + Style.RESET_ALL)


class ViLMA:
    """
    Main application class that coordinates the components.
    This class serves as the primary entry point and integrates
    all the modular components.
    """

    def __init__(self):
        """
        Initialize the ViLMA application by creating and connecting
        all the component modules.
        """
        # Initialize the component modules
        self.model_manager = ModelManager()
        self.screen_capture = ScreenCapture()
        self.action_handler = ActionHandler(self.screen_capture)
        self.config_manager = ConfigManager()
        self.ui = UserInterface(
            self.model_manager,
            self.screen_capture,
            self.action_handler,
            self.config_manager
        )

        logger.info("ViLMA initialized")

    def run(self):
        """
        Start the ViLMA application by displaying the user interface.
        """
        try:
            self.ui.terminal_menu()
        except Exception as e:
            logger.error(f"Error running ViLMA: {e}")
        finally:
            # Ensure proper cleanup
            self.action_handler.ensure_blank_window_closed()
            if self.screen_capture.is_recording:
                self.screen_capture.stop_recording()
            logger.info("ViLMA exited")


if __name__ == "__main__":
    try:
        # Create and run the ViLMA application
        vilma = ViLMA()
        print(Fore.CYAN + "Starting ViLMA (Vision-Language Model-based Active Monitoring)." + Style.RESET_ALL)
        vilma.run()
    except Exception as e:
        # If something fails on initialization, log it here
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Error initializing ViLMA: {e}")
        print(Fore.RED + f"{timestamp} - Error initializing ViLMA: {e}" + Style.RESET_ALL)
