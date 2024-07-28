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

init()

class ViLMA:
    """
    A class to monitor the desktop screen and perform binary inference on the captured images
    using a pre-trained vision-language model.

    Attributes:
        device (torch.device): The device to run the model on (CPU or CUDA).
        model (AutoModelForCausalLM): The pre-trained vision-language model.
        processor (AutoProcessor): The processor for the pre-trained model.
        prompts (list): List of prompts for binary inference.
        blank_window_open (bool): Flag to check if the blank window is open.
        logout_on_trigger (bool): Flag to check if the system should log out on 'YES' inference.
        dummy_mode (bool): Flag to check if the system is in dummy mode.
        full_output_mode (bool): Flag to check if the system should allow full model output.
        inference_rate (int or None): The number of inferences per second. Default is None.
    """

    def __init__(self):
        """
        Initializes the ViLMA without loading the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.prompts = []
        self.blank_window_open = False
        self.logout_on_trigger = False
        self.dummy_mode = False
        self.full_output_mode = False
        self.inference_rate = None

        atexit.register(self.ensure_blank_window_closed)

    def load_model(self, model_path):
        """
        Loads the model and processor with the given model path.

        Args:
            model_path (str): The path to the pre-trained model.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device).half()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print(Fore.GREEN + "Model loaded successfully." + Style.RESET_ALL)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def prepare_inputs(self, task_prompt, image):
        """
        Prepares inputs for the model.

        Args:
            task_prompt (str): The task prompt for the model.
            image (PIL.Image): The image to be processed.

        Returns:
            dict: The prepared inputs.
        """
        try:
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.half()
            return inputs
        except Exception as e:
            raise RuntimeError(f"Error preparing inputs: {e}")

    def run_model(self, inputs):
        """
        Runs the model on the prepared inputs.

        Args:
            inputs (dict): The prepared inputs.

        Returns:
            torch.Tensor: The generated IDs from the model.
        """
        try:
            with torch.amp.autocast("cuda"):
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )
            return generated_ids
        except Exception as e:
            raise RuntimeError(f"Error running model: {e}")

    def process_outputs(self, generated_ids):
        """
        Processes the outputs from the model.

        Args:
            generated_ids (torch.Tensor): The generated IDs from the model.

        Returns:
            str: The generated text from the model.
        """
        try:
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Error processing outputs: {e}")

    def run_inference(self, image, prompt):
        """
        Runs inference on the given image with the specified prompt.

        Args:
            image (PIL.Image): The image to be processed.
            prompt (str): The prompt for binary inference.

        Returns:
            bool: The result of the inference (True for 'yes', False for 'no').
        """
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>" if not self.full_output_mode else prompt
            inputs = self.prepare_inputs(task_prompt, image)
            generated_ids = self.run_model(inputs)
            generated_text = self.process_outputs(generated_ids)

            result = "yes" in generated_text.lower()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Prompt: {prompt} - Inference result: {'Yes' if result else 'No'} - {generated_text}")
            return result
        except Exception as e:
            print(f"Error during inference: {e}")
            return False

    def capture_desktop(self):
        """
        Captures the current desktop screen.

        Returns:
            PIL.Image: The captured desktop image.
        """
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
        except Exception as e:
            raise RuntimeError(f"Error capturing desktop: {e}")

    def show_blank_window(self):
        """
        Shows a blank window to obscure the screen.
        """
        try:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Showing blank window")
            self.blank_window_open = True
            blank_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.namedWindow("Blank Screen", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Blank Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty("Blank Screen", cv2.WND_PROP_TOPMOST, 1)
            while self.blank_window_open:
                cv2.imshow("Blank Screen", blank_screen)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.blank_window_open = False
                    break
            cv2.destroyWindow("Blank Screen")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Closed blank window")
        except Exception as e:
            print(f"Error showing blank window: {e}")

    def ensure_blank_window_closed(self):
        """
        Ensures the blank window is closed.
        """
        try:
            self.blank_window_open = False
            cv2.destroyWindow("Blank Screen")
        except:
            pass

    def logout(self):
        """
        Logs the user out of the system.
        """
        try:
            system_platform = platform.system()
            if system_platform == "Windows":
                # Command to log out on Windows
                subprocess.run(["shutdown", "/l"], check=True)
            elif system_platform == "Linux" or system_platform == "Darwin":
                # Command to log out on Unix-based systems (Linux/macOS)
                subprocess.run(["pkill", "-KILL", "-u", os.getlogin()], check=True)
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Unsupported operating system: {system_platform}")
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Error logging out: {e}")

    def start_monitoring(self):
        """
        Starts monitoring the desktop screen based on the configured prompts.
        """
        try:
            while True:
                start_time = time.time()
                screen = self.capture_desktop()
                screen_rgb = screen.convert("RGB")
                screen_np = np.array(screen_rgb)
                screen_resized = cv2.resize(screen_np, (640, 360))
                pil_image = Image.fromarray(screen_resized)

                if any(self.run_inference(pil_image, prompt) for prompt in self.prompts):
                    if self.dummy_mode:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, but dummy mode is ON")
                    else:
                        if self.logout_on_trigger:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, logging out")
                            self.logout()
                            break
                        if not self.blank_window_open:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, opening blank window")
                            self.show_blank_window()
                else:
                    if self.blank_window_open:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - No trigger detected, closing blank window")
                        self.ensure_blank_window_closed()

                elapsed_time = time.time() - start_time
                if self.inference_rate:
                    time_to_wait = max(1.0 / self.inference_rate - elapsed_time, 0)
                    time.sleep(time_to_wait)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Error during monitoring: {e}")
            self.ensure_blank_window_closed()

    def terminal_menu(self):
        """
        Displays the terminal menu for user interaction.
        """
        print(Fore.CYAN + "\n=== Welcome to ViLMA (Vision-Language Model-based Active Monitoring) ===" + Style.RESET_ALL)
        while True:
            print(Fore.CYAN + "\n=== ViLMA Terminal Menu ===" + Style.RESET_ALL)
            print(Fore.YELLOW + "Model Operations:" + Style.RESET_ALL)
            print(Fore.YELLOW + "1. Load Florence-2" + Style.RESET_ALL)
            print(Fore.YELLOW + "Monitoring Settings:" + Style.RESET_ALL)
            print(Fore.YELLOW + "2. Add Inference Prompt" + Style.RESET_ALL)
            print(Fore.YELLOW + "3. Remove Inference Prompt" + Style.RESET_ALL)
            print(Fore.YELLOW + "4. List Inference Prompts" + Style.RESET_ALL)
            print(Fore.YELLOW + "5. Set Inference Rate (current: " + (Fore.GREEN + str(self.inference_rate) if self.inference_rate else Fore.RED + "None") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.YELLOW + "Monitoring Control:" + Style.RESET_ALL)
            print(Fore.YELLOW + "6. Start Screen Monitoring" + Style.RESET_ALL)
            print(Fore.YELLOW + "7. Toggle Logout on Trigger (current: " + (Fore.GREEN + "ON" if self.logout_on_trigger else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.YELLOW + "8. Toggle Dummy Mode (current: " + (Fore.GREEN + "ON" if self.dummy_mode else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.YELLOW + "9. Toggle Full Output Mode (current: " + (Fore.GREEN + "ON" if self.full_output_mode else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.YELLOW + "General:" + Style.RESET_ALL)
            print(Fore.YELLOW + "10. Quit" + Style.RESET_ALL)
            print(Fore.CYAN + "==========================" + Style.RESET_ALL)
            choice = input("Enter your choice: ")

            try:
                if choice == "1":
                    self.load_model_menu()
                elif choice == "2":
                    prompt = input("Enter the inference prompt to add: ")
                    self.prompts.append(prompt)
                    print(Fore.GREEN + f"Added inference prompt: {prompt}" + Style.RESET_ALL)
                elif choice == "3":
                    self.list_prompts()
                    index = int(input("Enter the prompt number to remove: ")) - 1
                    if 0 <= index < len(self.prompts):
                        removed_prompt = self.prompts.pop(index)
                        print(Fore.GREEN + f"Removed inference prompt: {removed_prompt}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Invalid prompt number." + Style.RESET_ALL)
                elif choice == "4":
                    self.list_prompts()
                elif choice == "5":
                    self.set_inference_rate()
                elif choice == "6":
                    print(Fore.CYAN + "Starting screen monitoring..." + Style.RESET_ALL)
                    self.start_monitoring()
                elif choice == "7":
                    self.logout_on_trigger = not self.logout_on_trigger
                    print(Fore.GREEN + "Logout on Trigger is now {}".format("ON" if self.logout_on_trigger else "OFF") + Style.RESET_ALL)
                elif choice == "8":
                    self.dummy_mode = not self.dummy_mode
                    print(Fore.GREEN + "Dummy mode is now {}".format("ON" if self.dummy_mode else "OFF") + Style.RESET_ALL)
                elif choice == "9":
                    self.full_output_mode = not self.full_output_mode
                    print(Fore.GREEN + "Full output mode is now {}".format("ON" if self.full_output_mode else "OFF") + Style.RESET_ALL)
                elif choice == "10":
                    print(Fore.CYAN + "Quitting..." + Style.RESET_ALL)
                    break
                else:
                    print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

    def load_model_menu(self):
        """
        Prompts the user to select a model directory and loads the model.
        """
        model_path = select_model_path()
        if not model_path:
            print(Fore.RED + "Model path selection cancelled." + Style.RESET_ALL)
        else:
            self.load_model(model_path)

    def set_inference_rate(self):
        """
        Sets the inference rate for image processing per second.
        """
        try:
            rate = input("Enter the desired inference rate (1-25, or None for default): ")
            if rate.lower() == 'none':
                self.inference_rate = None
                print(Fore.GREEN + "Inference rate set to default (None)." + Style.RESET_ALL)
            else:
                rate = int(rate)
                if 1 <= rate <= 25:
                    self.inference_rate = rate
                    print(Fore.GREEN + f"Inference rate set to {rate} IPS." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid rate. Please enter a number between 1 and 25." + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a number between 1 and 25, or 'None'." + Style.RESET_ALL)

    def list_prompts(self):
        """
        Lists the current prompts.
        """
        print(Fore.CYAN + "\nCurrent inference prompts:" + Style.RESET_ALL)
        for i, prompt in enumerate(self.prompts, 1):
            print(Fore.GREEN + f"{i}. {prompt}" + Style.RESET_ALL)

def select_model_path():
    """
    Prompts the user to select a model directory.

    Returns:
        str: The selected model directory path.
    """
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askdirectory(title="Select Model Directory")
    root.destroy()
    return model_path

if __name__ == "__main__":
    try:
        vilma = ViLMA()
        print(Fore.CYAN + "Starting ViLMA (Vision-Language Model-based Active Monitoring)." + Style.RESET_ALL)
        vilma.terminal_menu()
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(Fore.RED + f"{timestamp} - Error initializing ViLMA: {e}" + Style.RESET_ALL)
