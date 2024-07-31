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

init()

class ViLMA:
    """
    A class to monitor the desktop screen and perform binary inference on the captured images
    using a pre-trained vision-language model.
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
        self.blank_screen_on_trigger = False
        self.screenshot_on_trigger = False
        self.record_on_trigger = False
        self.custom_trigger_path = None
        self.custom_trigger_enabled = False
        self.custom_trigger_output = "yes"
        self.recording = False
        self.inference_rate = None
        self.resolution = "720p"
        self.video_writer = None
        self.keyboard_trigger_enabled = False
        self.keyboard_trigger_sequence = ""
        self.keyboard_trigger_activated = False
        self.keyboard_trigger_output = "yes"

        atexit.register(self.ensure_blank_window_closed)

    def load_model(self, model_path):
        """
        Loads the model and processor with the given model path.
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
        """
        try:
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Error processing outputs: {e}")

    def run_inference(self, image, prompt):
        """
        Runs inference on the given image with the specified prompt.
        """
        try:
            task_prompt = prompt
            inputs = self.prepare_inputs(task_prompt, image)
            generated_ids = self.run_model(inputs)
            generated_text = self.process_outputs(generated_ids)

            cleaned_text = generated_text.replace("</s><s>", "").strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"{timestamp} - Raw Inference result: {generated_text}")
            print(f"{timestamp} - Cleaned Inference result: {cleaned_text}")

            if self.custom_trigger_enabled and self.custom_trigger_path:
                print(f"{timestamp} - Checking custom trigger: '{self.custom_trigger_output.lower()}' in '{cleaned_text.lower()}'")
                if self.custom_trigger_output.lower() in cleaned_text.lower():
                    print(f"{timestamp} - Custom trigger matched. Running custom trigger.")
                    self.run_custom_trigger()
                    self.custom_trigger_enabled = False
                else:
                    print(f"{timestamp} - Custom trigger did not match.")

            if self.keyboard_trigger_enabled and self.keyboard_trigger_output.lower() in cleaned_text.lower() and not self.keyboard_trigger_activated:
                print(f"{timestamp} - Keyboard trigger matched. Running keyboard trigger.")
                self.run_keyboard_trigger()
                self.keyboard_trigger_activated = True

            if self.keyboard_trigger_output.lower() not in cleaned_text.lower():
                self.keyboard_trigger_activated = False

            return cleaned_text
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Error"

    def capture_desktop(self):
        """
        Captures the current desktop screen.
        """
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
        except Exception as e:
            raise RuntimeError(f"Error capturing desktop: {e}")

    def take_screenshot(self):
        """
        Takes a screenshot of the current desktop.
        """
        try:
            screenshot = self.capture_desktop()
            screenshot.save(f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            print(Fore.GREEN + "Screenshot taken." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error taking screenshot: {e}" + Style.RESET_ALL)

    def start_recording(self):
        """
        Starts recording the desktop.
        """
        try:
            print(Fore.GREEN + "Recording started." + Style.RESET_ALL)
            self.recording = True
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width, height = self.get_resolution_dimensions()
            self.video_writer = cv2.VideoWriter(f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4', fourcc, 20.0, (width, height))

            threading.Thread(target=self.record_desktop).start()
        except Exception as e:
            print(Fore.RED + f"Error starting recording: {e}" + Style.RESET_ALL)

    def record_desktop(self):
        """
        Records the desktop screen.
        """
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while self.recording:
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.video_writer.write(frame)
                    time.sleep(1/20)
        except Exception as e:
            print(Fore.RED + f"Error recording desktop: {e}" + Style.RESET_ALL)

    def stop_recording(self):
        """
        Stops recording the desktop.
        """
        try:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            print(Fore.GREEN + "Recording stopped." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error stopping recording: {e}" + Style.RESET_ALL)

    def run_custom_trigger(self):
        """
        Runs the custom trigger command.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Running custom trigger: {self.custom_trigger_path}")
            if platform.system() == "Windows":
                os.startfile(self.custom_trigger_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", self.custom_trigger_path], check=True)
            else:
                subprocess.run(["xdg-open", self.custom_trigger_path], check=True)
            print(f"{timestamp} - Custom trigger executed successfully.")
        except Exception as e:
            print(f"{timestamp} - Error running custom trigger: {e}")

    def run_keyboard_trigger(self):
        """
        Runs the keyboard trigger command.
        """
        try:
            pyautogui.typewrite(self.keyboard_trigger_sequence)
            print(Fore.GREEN + f"Executed keyboard sequence: {self.keyboard_trigger_sequence}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error executing keyboard trigger: {e}" + Style.RESET_ALL)

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

            if platform.system() == "Windows":
                hwnd = ctypes.windll.user32.FindWindowW(None, "Blank Screen")
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

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
                subprocess.run(["shutdown", "/l"], check=True)
            elif system_platform == "Linux" or system_platform == "Darwin":
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
        if self.model is None:
            print(Fore.RED + "Error: No model loaded. Please load a model before starting monitoring." + Style.RESET_ALL)
            return

        if not self.prompts:
            print(Fore.RED + "Error: No inference prompts set. Please add at least one inference prompt before starting monitoring." + Style.RESET_ALL)
            return

        try:
            while True:
                start_time = time.time()
                screen = self.capture_desktop()
                screen_rgb = screen.convert("RGB")
                screen_np = np.array(screen_rgb)
                width, height = self.get_resolution_dimensions()
                screen_resized = cv2.resize(screen_np, (width, height))
                pil_image = Image.fromarray(screen_resized)

                for prompt in self.prompts:
                    result = self.run_inference(pil_image, prompt)
                    if result.lower() == "yes" and not self.dummy_mode:
                        if self.logout_on_trigger:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, logging out")
                            self.logout()
                            break
                        if self.blank_screen_on_trigger and not self.blank_window_open:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, opening blank window")
                            self.show_blank_window()
                    elif result.lower() == "no" and self.blank_window_open:
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

    def get_resolution_dimensions(self):
        """
        Returns the dimensions based on the selected resolution.
        """
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
            return 1280, 720

    def toggle_resolution(self):
        """
        Toggles the resolution for image processing.
        """
        resolutions = ["640p", "720p", "1080p", "native"]
        current_index = resolutions.index(self.resolution)
        new_index = (current_index + 1) % len(resolutions)
        self.resolution = resolutions[new_index]
        print(Fore.GREEN + f"Resolution set to {self.resolution}." + Style.RESET_ALL)

    def terminal_menu(self):
        """
        Displays the terminal menu for user interaction.
        """
        print(Fore.CYAN + "\n=== Welcome to ViLMA (Vision-Language Model-based Active Monitoring) ===" + Style.RESET_ALL)
        while True:
            print(Fore.CYAN + "\n=== Menu ===" + Style.RESET_ALL)

            print(Fore.LIGHTGREEN_EX + "1. Start Screen Monitoring" + Style.RESET_ALL)

            print(Fore.MAGENTA + "\nModel Operations:" + Style.RESET_ALL)
            print(Fore.LIGHTMAGENTA_EX + "2. Load Florence-2" + Style.RESET_ALL)

            print(Fore.BLUE + "\nMonitoring Settings:" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + "3. Add Inference Prompt" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + "4. Remove Inference Prompt" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + "5. List Inference Prompts" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + "6. Set Inference Rate (current: " + (Fore.GREEN + str(self.inference_rate) if self.inference_rate else Fore.RED + "None") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + "7. Toggle Processing Resolution (current: " + Fore.GREEN + self.resolution + Style.RESET_ALL + ")" + Style.RESET_ALL)

            print(Fore.GREEN + "\nMonitoring Control:" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "8. Toggle Logout on Trigger (current: " + (Fore.GREEN + "ON" if self.logout_on_trigger else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "9. Toggle Dummy Mode (current: " + (Fore.GREEN + "ON" if self.dummy_mode else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "10. Toggle Blank Screen on Trigger (current: " + (Fore.GREEN + "ON" if self.blank_screen_on_trigger else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "11. Toggle Screenshot on Trigger (current: " + (Fore.GREEN + "ON" if self.screenshot_on_trigger else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "12. Toggle Record on Trigger (current: " + (Fore.GREEN + "ON" if self.record_on_trigger else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "13. Toggle Custom Trigger (current: " + (Fore.GREEN + "ON" if self.custom_trigger_enabled else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)
            print(Fore.LIGHTGREEN_EX + "14. Toggle Keyboard Command on Trigger (current: " + (Fore.GREEN + "ON" if self.keyboard_trigger_enabled else Fore.RED + "OFF") + Style.RESET_ALL + ")" + Style.RESET_ALL)

            print(Fore.YELLOW + "\nGeneral:" + Style.RESET_ALL)
            print(Fore.LIGHTYELLOW_EX + "15. Quit" + Style.RESET_ALL)

            print(Fore.CYAN + "\n==========================" + Style.RESET_ALL)
            choice = input("Enter your choice: ")

            try:
                if choice == "1":
                    print(Fore.CYAN + "Starting screen monitoring..." + Style.RESET_ALL)
                    self.start_monitoring()
                elif choice == "2":
                    self.load_model_menu()
                elif choice == "3":
                    prompt = input("Enter the inference prompt to add: ")
                    self.prompts.append(prompt)
                    print(Fore.GREEN + f"Added inference prompt: {prompt}" + Style.RESET_ALL)
                elif choice == "4":
                    self.list_prompts()
                    index = int(input("Enter the prompt number to remove: ")) - 1
                    if 0 <= index < len(self.prompts):
                        removed_prompt = self.prompts.pop(index)
                        print(Fore.GREEN + f"Removed inference prompt: {removed_prompt}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Invalid prompt number." + Style.RESET_ALL)
                elif choice == "5":
                    self.list_prompts()
                elif choice == "6":
                    self.set_inference_rate()
                elif choice == "7":
                    self.toggle_resolution()
                elif choice == "8":
                    self.logout_on_trigger = not self.logout_on_trigger
                    print(Fore.GREEN + "Logout on Trigger is now {}".format("ON" if self.logout_on_trigger else "OFF") + Style.RESET_ALL)
                elif choice == "9":
                    self.dummy_mode = not self.dummy_mode
                    print(Fore.GREEN + "Dummy mode is now {}".format("ON" if self.dummy_mode else "OFF") + Style.RESET_ALL)
                elif choice == "10":
                    self.blank_screen_on_trigger = not self.blank_screen_on_trigger
                    print(Fore.GREEN + "Blank Screen on Trigger is now {}".format("ON" if self.blank_screen_on_trigger else "OFF") + Style.RESET_ALL)
                elif choice == "11":
                    self.screenshot_on_trigger = not self.screenshot_on_trigger
                    print(Fore.GREEN + "Screenshot on Trigger is now {}".format("ON" if self.screenshot_on_trigger else "OFF") + Style.RESET_ALL)
                elif choice == "12":
                    self.record_on_trigger = not self.record_on_trigger
                    print(Fore.GREEN + "Record on Trigger is now {}".format("ON" if self.record_on_trigger else "OFF") + Style.RESET_ALL)
                elif choice == "13":
                    if not self.custom_trigger_enabled:
                        self.custom_trigger_path = filedialog.askopenfilename(title="Select File to Open on Trigger")
                        if self.custom_trigger_path:
                            self.custom_trigger_output = input("Enter the output that triggers the custom action (e.g., 'yes', 'no', 'open file', etc.): ")
                            print(f"Setting custom_trigger_output to: {self.custom_trigger_output}")
                            self.custom_trigger_enabled = True
                            print(Fore.GREEN + f"Custom Trigger set to open {self.custom_trigger_path} on output: {self.custom_trigger_output}" + Style.RESET_ALL)
                        else:
                            print(Fore.RED + "Custom Trigger path selection cancelled." + Style.RESET_ALL)
                    else:
                        self.custom_trigger_enabled = False
                        self.custom_trigger_path = None
                        self.custom_trigger_output = "yes"
                        print(Fore.GREEN + "Custom Trigger is now OFF" + Style.RESET_ALL)
                elif choice == "14":
                    if not self.keyboard_trigger_enabled:
                        self.keyboard_trigger_sequence = input("Enter the keyboard sequence to type on trigger: ")
                        self.keyboard_trigger_output = input("Enter the output that triggers the keyboard action (e.g., 'yes', 'no', 'low health', etc.): ")
                        self.keyboard_trigger_enabled = True
                        self.keyboard_trigger_activated = False
                        print(Fore.GREEN + f"Keyboard Trigger set to type: {self.keyboard_trigger_sequence} on output: {self.keyboard_trigger_output}" + Style.RESET_ALL)
                    else:
                        self.keyboard_trigger_enabled = False
                        self.keyboard_trigger_sequence = ""
                        self.keyboard_trigger_output = "yes"
                        print(Fore.GREEN + "Keyboard Trigger is now OFF" + Style.RESET_ALL)
                elif choice == "15":
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
