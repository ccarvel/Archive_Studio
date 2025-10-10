import os, cv2, shutil, threading
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import math
from PIL import Image, ImageTk, ImageDraw, ImageChops
import pandas as pd
import concurrent.futures

class ThresholdAdjuster(tk.Toplevel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.title("Threshold Adjustment")
        self.parent = parent
        self.image_path = image_path
        
        # Initialize variables
        self.threshold_var = tk.IntVar(value=127)
        self.margin_var = tk.IntVar(value=10)
        
        # Load the image
        self.original_image = cv2.imread(image_path)
        # --- ADDED: Check if image loaded successfully ---
        if self.original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {os.path.basename(image_path)}\\nPlease ensure the file exists and is a valid image.", parent=self)
            self.destroy() # Close the adjuster window
            return # Stop initialization
        # --- END ADDED ---
        self.height, self.width = self.original_image.shape[:2]
        
        # Calculate preview scale to ensure window isn't too tall
        screen_height = self.winfo_screenheight()
        max_preview_height = screen_height * 0.6  # Use 60% of screen height as maximum
        self.preview_scale = min(0.5, max_preview_height / self.height)  # Use 0.5 or smaller if needed
        
        # Calculate preview dimensions
        self.preview_width = int(self.width * self.preview_scale)
        self.preview_height = int(self.height * self.preview_scale)
        
        # Create the GUI
        self.create_widgets()
        self.update_preview()
        
        # Make window modal
        self.transient(parent)
        self.grab_set()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            main_frame, 
            width=self.preview_width,
            height=self.preview_height
        )
        self.preview_canvas.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Threshold control
        ttk.Label(main_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=lambda _: self.update_preview()
        )
        threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Margin control
        ttk.Label(main_frame, text="Margin:").grid(row=2, column=0, sticky=tk.W, pady=5)
        margin_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.margin_var,
            command=lambda _: self.update_preview()
        )
        margin_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_crop).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

        # Configure column weights
        main_frame.columnconfigure(1, weight=1)

    def update_preview(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create preview image
        preview = self.original_image.copy()
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw the contour and crop rectangle
            cv2.drawContours(preview, [largest_contour], -1, (0, 255, 0), 2)
            
            # Get bounding rectangle with margin
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = self.margin_var.get()
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(self.width - x, w + 2 * margin)
            h = min(self.height - y, h + 2 * margin)
            
            # Draw crop rectangle
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Resize for preview
        preview = cv2.resize(preview, (self.preview_width, self.preview_height))
        
        # Convert to PhotoImage
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview)
        self.preview_photo = ImageTk.PhotoImage(preview)
        
        # Update canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)

    def apply_crop(self):
        # Store the values in parent
        self.parent.threshold_value = self.threshold_var.get()
        self.parent.margin_value = self.margin_var.get()
        
        # Apply the crop
        self.parent.crop_to_largest_white_area(
            self.image_path,
            threshold=self.threshold_var.get(),
            margin=self.margin_var.get()
        )
        
        # Close the window
        self.destroy()

class ImageSplitter(tk.Toplevel):
    def __init__(self, active_directory):
        super().__init__()
        self.state('zoomed')
        self.title("Transcription Pearl Image Preprocessing Tool 0.9 beta")
        
        # Store the directory path, but validate it
        if not active_directory:
            messagebox.showerror("Error", "No directory specified for image editing.")
            self.destroy()
            return
        
        if not os.path.exists(active_directory):
            try:
                os.makedirs(active_directory, exist_ok=True)
                print(f"Created directory: {active_directory}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create directory: {active_directory}\n{str(e)}")
                self.destroy()
                return
            
        self.folder_path = active_directory
        self.link_nav = 0
        self.current_image_index = 0
       
        self.special_cursor_active = False
        
        self.split_line = None
        self.split_start = None
        self.split_end = None
        self.current_scale = 1.0
        self.original_image = None
        self.cursor_orientation = 'vertical'  # Can be 'vertical' or 'horizontal'

        # Initialize auto_split before creating menus
        self.auto_split = False
        self.auto_split_var = tk.BooleanVar(value=False)

        # Determines if the image is cropped on release of the mouse (True) or on press of Enter (False)
        self.batch_process = tk.BooleanVar()
        self.batch_process.set(False)

        # Create the GUI elements
        self.create_widgets()
        self.create_menus()
        self.create_key_bindings()
        
        # Load images after a short delay to ensure window is ready
        self.after(100, self.load_a_folder)

        self.crop_rect = None
        self.crop_start = None
        self.crop_end = None
        self.cropping = False
        
        self.angled_cursor_active = False
        self.cursor_angle = 0
        self.cursor_line = None
        self.vertical_line = None
        self.horizontal_line = None
        self.special_cursor_active = False
        self.horizontal_cursor_active = False
        
        self.auto_split = False

        self.image_data = pd.DataFrame(columns=['Image_Index', 'Original_Image', 'Split_Image', 'Left_or_Right'])

        self.status = "no_changes"  # Possible values: "no_changes", "saved", "discarded"
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.geometry("800x600")

        # Main frame to hold canvas and buttons
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.image_canvas = tk.Canvas(main_frame, highlightthickness=0)
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Navigation frame
        self.navigation_frame = tk.Frame(main_frame)
        self.navigation_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.cursor_angle = 0
        self.cursor_line = None

        # Quick Crop Checkbox
        self.batch_process_check = tk.Checkbutton(self.navigation_frame, text="Batch Process", variable=self.batch_process)
        self.batch_process_check.pack(side=tk.LEFT, padx=5)

        self.rotate_left_button = tk.Button(self.navigation_frame, text="↻", command=lambda: self.rotate_image(-90))
        self.rotate_left_button.pack(side=tk.LEFT, padx=5)

        self.rotate_right_button = tk.Button(self.navigation_frame, text="↺", command=lambda: self.rotate_image(90))
        self.rotate_right_button.pack(side=tk.LEFT, padx=5)

        self.first_button = tk.Button(self.navigation_frame, text="|<", command=lambda: self.navigate_images(-2))
        self.first_button.pack(side=tk.LEFT, padx=5)

        self.prev_button = tk.Button(self.navigation_frame, text="<--", command=lambda: self.navigate_images(-1))
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.navigation_frame, text="-->", command=lambda: self.navigate_images(1))
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.last_button = tk.Button(self.navigation_frame, text=">|", command=lambda: self.navigate_images(2))
        self.last_button.pack(side=tk.LEFT, padx=5)

        self.image_canvas.bind("<Motion>", self.update_cursor_line)
        
    def create_menus(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_split_images)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Revert Current Image", command=self.revert_to_original)
        edit_menu.add_command(label="Revert All Images", command=self.revert_all_images)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Current Image", command=self.delete_current_image)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="First Image", command=lambda: self.navigate_images(-2))
        view_menu.add_command(label="Back Image", command=lambda: self.navigate_images(-1))
        view_menu.add_command(label="Forward Image", command=lambda: self.navigate_images(1))
        view_menu.add_command(label="Last Image", command=lambda: self.navigate_images(2))
        menubar.add_cascade(label="View", menu=view_menu)

        # Process menu
        process_menu = tk.Menu(menubar, tearoff=0)
        process_menu.add_command(label="Split Image", command=lambda: self.switch_to_vertical())
        self.auto_split_var = tk.BooleanVar(value=False)
        process_menu.add_checkbutton(label="Split All Images", 
                                variable=self.auto_split_var,
                                command=self.toggle_auto_split)
        process_menu.add_separator()
        process_menu.add_command(label="Crop Image", command=self.activate_crop_tool)
        process_menu.add_separator()
        process_menu.add_command(label="Auto Crop Active Image", command=self.auto_crop_image)
        process_menu.add_command(label="Auto Crop All Images", command=self.auto_crop_all_images)
        process_menu.add_separator()
        process_menu.add_command(label="Auto Straighten Image", command=self.auto_straighten_image)
        process_menu.add_command(label="Auto Straighten All Images", command=self.auto_straighten_all_images)
        process_menu.add_separator()
        process_menu.add_command(label="Straighten Image by Line", command=self.manual_straighten)
        process_menu.add_separator()
        process_menu.add_command(label="Rotate Image Clockwise", command=lambda: self.rotate_image(-90))
        process_menu.add_command(label="Rotate Image Counter-Clockwise", command=lambda: self.rotate_image(90))
        process_menu.add_separator()
        process_menu.add_command(label="Rotate Image by Angle", command=self.incremental_rotate)  # Add this line
        process_menu.add_separator()
        process_menu.add_command(label="Rotate All Images Clockwise", command=lambda: self.rotate_all_images(-90))
        process_menu.add_command(label="Rotate All Images Counter-Clockwise", command=lambda: self.rotate_all_images(90))        
        menubar.add_cascade(label="Process", menu=process_menu)       

    def create_key_bindings(self):
        # Update cursor bindings
        self.bind("<Control-h>", self.switch_to_horizontal) 
        self.bind("<Control-v>", self.switch_to_vertical)
        self.bind("<Control-a>", self.toggle_auto_split)
        
        # Straighten image by line
        self.bind("<Control-l>", self.manual_straighten)

        # Crop image
        self.bind("<Control-Shift-c>", self.activate_crop_tool)

        # Auto crop image
        self.bind("<Control-Shift-a>", self.auto_crop_image)

        # Auto straighten
        self.bind("<Control-s>", self.auto_straighten_image)
        self.bind("<Control-Shift-s>", self.auto_straighten_all_images)

        # Update rotation bindings for cursor
        self.bind("<bracketright>", lambda e: self.rotate_cursor(-1))
        self.bind("<bracketleft>", lambda e: self.rotate_cursor(1))

        # Mouse and navigation bindings
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        self.image_canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.bind("<Left>", lambda event: self.navigate_images(-1))
        self.bind("<Right>", lambda event: self.navigate_images(1))
        
        # Ctrl+Arrow navigation bindings
        self.bind("<Control-Left>", lambda event: self.navigate_images(-1))
        self.bind("<Control-Right>", lambda event: self.navigate_images(1))

        # Image rotation bindings
        self.bind("<Control-bracketright>", lambda event: self.rotate_image(-90))
        self.bind("<Control-bracketleft>", lambda event: self.rotate_image(90))
        self.bind("<Control-Shift-bracketright>", lambda event: self.rotate_all_images(-90))
        self.bind("<Control-Shift-bracketleft>", lambda event: self.rotate_all_images(90))

# Loading functions
    
    def load_a_folder(self):
        if self.folder_path:
            # Set up temp directory
            self.set_temp_dir()
            
            # Look for image files in the temp directory
            image_files = [file for file in os.listdir(self.temp_folder) if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"))]
            
            # Handle case where no images are found
            if not image_files:
                print(f"No images found in temp folder: {self.temp_folder}")
                # If editing a single file (sent from the main app), try to create a blank image placeholder
                try:
                    # Create a simple blank image as a placeholder
                    blank_image = Image.new('RGB', (800, 1000), (255, 255, 255))  # White background
                    blank_path = os.path.join(self.temp_folder, "blank_image.jpg")
                    blank_image.save(blank_path, 'JPEG', quality=98, optimize=True)
                    image_files = ["blank_image.jpg"]
                    print(f"Created placeholder image at {blank_path}")
                except Exception as e:
                    print(f"Error creating placeholder image: {e}")
                    messagebox.showwarning("No Images", "No images found to edit. The tool will close.")
                    self.after(500, self.destroy)
                    return
            
            # Initialize DataFrame
            self.image_data = pd.DataFrame(columns=['Image_Index', 'Original_Image', 'Split_Image', 'Left_or_Right'])
            
            # Add image files to DataFrame
            for i, image_file in enumerate(image_files, start=1):
                image_path = os.path.join(self.temp_folder, image_file)
                self.image_data = pd.concat([self.image_data, pd.DataFrame({
                    'Image_Index': [i], 
                    'Original_Image': [image_path], 
                    'Split_Image': [None], 
                    'Left_or_Right': [None]
                })], ignore_index=True)
            
            if not self.image_data.empty:
                self.current_image_index = 0
                self.show_current_image()
            else:
                messagebox.showwarning("No Images", "No valid images could be loaded. The tool will close.")
                self.after(500, self.destroy)
    
    def process_split_image(self, current_image_path, current_image_row, left_image, right_image, split_type):
        try:
            # Get current sequence number from the filename
            # Handle the new format "0001_p001" by splitting on '_' and taking first part
            base_name = os.path.splitext(os.path.basename(current_image_path))[0]
            
            # Check different filename patterns
            if '_p' in base_name:
                current_seq = int(base_name.split('_p')[0])
            else:
                # Try to extract any numeric part for sequence number
                try:
                    current_seq = int(''.join(filter(str.isdigit, base_name)))
                except ValueError:
                    # If all else fails, use current_image_index + 1
                    current_seq = self.current_image_index + 1
            
            # Calculate new sequence numbers for split images
            next_images = [f for f in os.listdir(os.path.dirname(current_image_path)) 
                        if f.endswith('.jpg') and f != os.path.basename(current_image_path)]
            
            # Extract sequence numbers from filenames, handling the new format
            existing_numbers = []
            for fname in next_images:
                try:
                    if '_p' in fname:
                        num = int(os.path.splitext(fname)[0].split('_p')[0])
                    else:
                        num = int(''.join(filter(str.isdigit, os.path.splitext(fname)[0])))
                    existing_numbers.append(num)
                except (ValueError, IndexError):
                    continue
                    
            next_seq = max([current_seq] + existing_numbers) + 1
            
            # Generate new file names with proper sequential numbering
            image_dir = os.path.dirname(current_image_path)
            left_image_path = os.path.join(image_dir, 
                                        f"{current_seq:04d}_p{current_seq:03d}.jpg")
            right_image_path = os.path.join(image_dir, 
                                        f"{next_seq:04d}_p{next_seq:03d}.jpg")

            # Ensure images are in RGB mode before saving
            left_image = left_image.convert("RGB")
            right_image = right_image.convert("RGB")
            
            # Save the split images with high quality
            left_image.save(left_image_path, 'JPEG', quality=98, optimize=True)
            right_image.save(right_image_path, 'JPEG', quality=98, optimize=True)
            
            # Get the current index in the DataFrame
            # Handle different types of indices safely
            if isinstance(current_image_row.name, (int, np.integer)):
                # Use the numeric index directly
                current_index = current_image_row.name
            else:
                # Fall back to current_image_index
                current_index = self.current_image_index
            
            # Create new rows for the split images
            left_row = pd.DataFrame({
                'Image_Index': [current_index + 1],
                'Original_Image': [current_image_path],
                'Split_Image': [left_image_path],
                'Left_or_Right': ['Left']
            })
            
            right_row = pd.DataFrame({
                'Image_Index': [next_seq],
                'Original_Image': [current_image_path],
                'Split_Image': [right_image_path],
                'Left_or_Right': ['Right']
            })

            # Remove the original row and insert the new split rows
            # Use a try-except block to handle potential indexing issues
            try:
                before_split = self.image_data.loc[:current_index-1] if current_index > 0 else pd.DataFrame()
                after_split = self.image_data.loc[current_index+1:] if current_index < len(self.image_data)-1 else pd.DataFrame()
            except Exception:
                # If there's any issue with indexing, use a different approach
                before_split = self.image_data[self.image_data['Image_Index'] < self.current_image_index + 1] 
                after_split = self.image_data[self.image_data['Image_Index'] > self.current_image_index + 1]
            
            self.image_data = pd.concat([
                before_split,
                left_row,
                right_row,
                after_split
            ]).reset_index(drop=True)
            
            # Update all Image_Index values to ensure proper sequence
            self.image_data['Image_Index'] = range(1, len(self.image_data) + 1)
            
            return True
            
        except Exception as e:
            print(f"Error in process_split_image: {str(e)}")
            return False
            
    def clear_all_modes(self):
        # Clear crop mode
        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        if self.crop_rect:
            self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None
        
        # Clear cursor lines
        self.clear_cursor_lines()
        
        # Reset cursor
        self.image_canvas.config(cursor="")
        
        # Unbind all relevant events
        self.image_canvas.unbind("<ButtonPress-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonRelease-1>")
        self.image_canvas.unbind("<Motion>")
        self.unbind("<Return>")
        self.unbind("<Escape>")
        
        # Reset splitting-related variables
        self.special_cursor_active = False
        self.cursor_angle = 0
        self.cursor_orientation = 'vertical'  # Default to vertical
        
        # Rebind the basic navigation controls
        self.bind("<Left>", lambda event: self.navigate_images(-1))
        self.bind("<Right>", lambda event: self.navigate_images(1))
        self.bind("<Control-Left>", lambda event: self.navigate_images(-1))
        self.bind("<Control-Right>", lambda event: self.navigate_images(1))
        self.bind("<KeyPress-,>", lambda event: self.rotate_image(-90))
        self.bind("<KeyPress-.>", lambda event: self.rotate_image(90))
        
        # Rebind splitting-related controls
        self.bind("<Control-h>", self.switch_to_horizontal)
        self.bind("<Control-v>", self.switch_to_vertical)

    def clear_cursor_lines(self):
        """Clears all cursor lines from the canvas"""
        if self.vertical_line:
            self.image_canvas.delete(self.vertical_line)
            self.vertical_line = None
        if self.horizontal_line:
            self.image_canvas.delete(self.horizontal_line)
            self.horizontal_line = None
        if self.cursor_line:
            self.image_canvas.delete(self.cursor_line)
            self.cursor_line = None

    def toggle_cursor(self, event=None):
        """Toggles the cursor on/off"""
        self.special_cursor_active = not self.special_cursor_active
        
        if self.special_cursor_active:
            # Default to vertical if no orientation is set
            if not hasattr(self, 'cursor_orientation'):
                self.cursor_orientation = 'vertical'
            self.cursor_angle = 0  # Reset angle when activating cursor
            self.image_canvas.config(cursor="none")
            self.image_canvas.bind("<Motion>", self.update_cursor_line)
            self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
            # Force an initial cursor line update
            mock_event = type('MockEvent', (), {
                'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
                'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
            })
            self.update_cursor_line(mock_event)
        else:
            self.image_canvas.config(cursor="")
            self.image_canvas.unbind("<Motion>")
            self.image_canvas.unbind("<Button-1>")
            self.clear_cursor_lines()
            
    def switch_to_vertical(self, event=None):
        """Explicitly switches to vertical cursor mode"""
        self.clear_all_modes()
        self.cursor_orientation = 'vertical'
        self.special_cursor_active = True
        self.image_canvas.config(cursor="none")
        
        # Bind the necessary events for splitting
        self.image_canvas.bind("<Motion>", self.update_cursor_line)
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        
        # Force cursor update with current mouse position
        mock_event = type('MockEvent', (), {
            'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
            'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
        })
        self.update_cursor_line(mock_event)

    def switch_to_horizontal(self, event=None):
        """Explicitly switches to horizontal cursor mode"""
        self.clear_all_modes()
        self.cursor_orientation = 'horizontal'
        self.special_cursor_active = True
        self.image_canvas.config(cursor="none")
        
        # Bind the necessary events for splitting
        self.image_canvas.bind("<Motion>", self.update_cursor_line)
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        
        # Force cursor update with current mouse position
        mock_event = type('MockEvent', (), {
            'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
            'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
        })
        self.update_cursor_line(mock_event)

    def update_cursor_line(self, event):
        if not self.special_cursor_active or not self.original_image:
            return

        # Clear existing lines
        self.clear_cursor_lines()
        
        # Get canvas and image dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Get the displayed image dimensions (scaled)
        image_width, image_height = self.original_image.size
        scaled_width = int(image_width * self.current_scale)
        scaled_height = int(image_height * self.current_scale)
        
        x = self.image_canvas.canvasx(event.x)
        y = self.image_canvas.canvasy(event.y)
        
        # Only draw cursor lines if mouse is within the image area
        if not (0 <= x <= scaled_width and 0 <= y <= scaled_height):
            return

        if self.cursor_orientation == 'angled':
            # Calculate line length that will extend across the entire image area (not canvas)
            line_length = math.sqrt(scaled_width**2 + scaled_height**2) * 2
            
            # Convert angle to radians
            angle_rad = math.radians(self.cursor_angle)
            
            # Calculate dx and dy for the line
            dx = line_length * math.cos(angle_rad)
            dy = line_length * math.sin(angle_rad)
            
            # Calculate line endpoints centered on mouse position
            x1 = x - dx/2
            y1 = y - dy/2
            x2 = x + dx/2
            y2 = y + dy/2
            
            self.cursor_line = self.image_canvas.create_line(
                x1, y1, x2, y2,
                fill="red", width=2
            )
        elif self.cursor_orientation == 'vertical':
            # Limit vertical line to image height
            self.vertical_line = self.image_canvas.create_line(
                x, 0, x, scaled_height,
                fill="red", width=2
            )
        else:  # horizontal
            # Limit horizontal line to image width
            self.horizontal_line = self.image_canvas.create_line(
                0, y, scaled_width, y,
                fill="red", width=2
            )

    def handle_mouse_click(self, event):
        if self.special_cursor_active and self.original_image:
            try:
                # Check if click is within the image bounds
                if not self.is_click_within_image(event):
                    return
                    
                # Get the current cursor coordinates
                if self.cursor_orientation == 'angled' and self.cursor_line:
                    coords = self.image_canvas.coords(self.cursor_line)
                    if not coords:
                        return
                elif self.cursor_orientation == 'vertical' and self.vertical_line:
                    coords = self.image_canvas.coords(self.vertical_line)
                    if not coords:
                        return
                elif self.cursor_orientation == 'horizontal' and self.horizontal_line:
                    coords = self.image_canvas.coords(self.horizontal_line)
                    if not coords:
                        return
                else:
                    return

                self.call_split_image_functions()
                self.clear_cursor_lines()
                
                if self.batch_process.get():
                    # Move two images ahead after splitting
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, lambda: self.navigate_images(1))
                    
                # Force cursor update with current mouse position
                mock_event = type('MockEvent', (), {
                    'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
                    'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
                })
                self.update_cursor_line(mock_event)
                    
            except Exception as e:
                print(f"Error in handle_mouse_click: {str(e)}")

    def is_click_within_image(self, event):
        """Check if the mouse click is within the displayed image bounds"""
        try:
            if not self.original_image:
                return False
                
            # Get click coordinates
            click_x = self.image_canvas.canvasx(event.x)
            click_y = self.image_canvas.canvasy(event.y)
            
            # Get the displayed image dimensions (scaled)
            image_width, image_height = self.original_image.size
            scaled_width = int(image_width * self.current_scale)
            scaled_height = int(image_height * self.current_scale)
            
            # Check if click is within the scaled image bounds
            # The image is placed at (0, 0) on the canvas
            if 0 <= click_x <= scaled_width and 0 <= click_y <= scaled_height:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error in is_click_within_image: {str(e)}")
            return False

    def on_threshold_margin_key_press(self, event):
        if event.keysym == 'space':
            return 'break'

    def toggle_auto_split(self):
        # Clear cursor modes when toggling auto split
        if hasattr(self, 'special_cursor_active') and self.special_cursor_active:
            self.clear_all_modes()
            
        self.auto_split = not self.auto_split

    def toggle_special_cursor(self):
        """Toggles the special cursor mode"""
        self.special_cursor_active = not self.special_cursor_active
        
        if self.special_cursor_active:
            self.image_canvas.config(cursor="none")
            self.image_canvas.bind("<Motion>", self.update_cursor_line)
            self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
            
            # Force an initial cursor line update
            mock_event = type('MockEvent', (), {
                'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
                'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
            })
            self.update_cursor_line(mock_event)
        else:
            self.image_canvas.config(cursor="")
            self.image_canvas.unbind("<Motion>")
            self.image_canvas.unbind("<Button-1>")
            self.clear_cursor_lines()

    def ensure_cursor_bindings(self):
        if self.special_cursor_active:
            self.image_canvas.bind("<Motion>", self.update_cursor_line)
            self.image_canvas.bind("<Button-1>", self.handle_mouse_click)

# Routing Functions

    def show_edge_detection(self):
        # Clear cursor modes when showing edge detection
        self.clear_all_modes()
        
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            ThresholdAdjuster(self, image_path)

    def call_split_image_functions(self):
        """Routes to the appropriate split function based on auto_split status"""
        if self.auto_split:
            self.split_all_images()
        else:
            self.split_image_manually()

# Auto Straighten Functions

    def manual_straighten(self):
        # Clear existing cursor modes before starting manual straighten
        self.clear_all_modes()
        
        if not self.image_data.empty:
            self.straightening_mode = True
            self.straighten_start = None
            self.straighten_line = None
            self.guide_line = None
            
            def on_click(event):
                if not self.straighten_start:
                    # First click - store starting point
                    self.straighten_start = (event.x, event.y)
                    # Bind motion for live line preview
                    self.image_canvas.bind('<Motion>', update_guide_line)
                else:
                    # Second click - calculate angle and rotate
                    end_point = (event.x, event.y)
                    calculate_and_rotate(self.straighten_start, end_point)
                    # Clean up and handle quick crop behavior
                    cleanup()
                    
                    # If quick crop is enabled, move to next image and reactivate straightening
                    if self.batch_process.get():
                        self.after(100, lambda: self.navigate_images(1))
                        self.after(200, self.manual_straighten)  # Reactivate straightening mode

            def update_guide_line(event):
                if self.straighten_start:
                    # Clear previous guide line
                    if self.guide_line:
                        self.image_canvas.delete(self.guide_line)
                    # Draw new guide line
                    self.guide_line = self.image_canvas.create_line(
                        self.straighten_start[0], self.straighten_start[1],
                        event.x, event.y,
                        fill='blue', width=2
                    )

            def calculate_and_rotate(start, end):
                # Calculate angle between points
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = math.degrees(math.atan2(dy, dx))

                # Determine if line is more vertical or horizontal
                is_vertical = abs(dx) < abs(dy)

                if is_vertical:
                    # For vertical lines, normalize angle to be relative to 90°
                    target_angle = 90
                    current_angle = angle % 180
                    if current_angle < 0:
                        current_angle += 180
                else:
                    # For horizontal lines, normalize angle to be relative to 0°
                    target_angle = 0
                    current_angle = angle % 180
                    if current_angle < 0:
                        current_angle += 180

                # Calculate the minimum rotation needed
                rotation_needed = target_angle - current_angle
                
                # Normalize rotation to smallest angle
                if abs(rotation_needed) > 90:
                    if rotation_needed > 0:
                        rotation_needed -= 180
                    else:
                        rotation_needed += 180

                # Rotate the image
                self.rotate_image(rotation_needed)

            def cleanup():
                # Only clear bindings and temporary elements if not in quick crop mode
                if not self.batch_process.get():
                    self.image_canvas.unbind('<Button-1>')
                    self.image_canvas.unbind('<Motion>')
                    if self.guide_line:
                        self.image_canvas.delete(self.guide_line)
                    self.straightening_mode = False
                    self.straighten_start = None
                    self.guide_line = None
                    self.image_canvas.config(cursor="")
                else:
                    # Just clear the current guide line and reset start point
                    if self.guide_line:
                        self.image_canvas.delete(self.guide_line)
                    self.straighten_start = None
                    self.guide_line = None

            # Set up the canvas for straightening
            self.image_canvas.config(cursor="crosshair")
            self.image_canvas.bind('<Button-1>', on_click)

# Crop Functions

    def crop_all_images(self):
        if not self.image_data.empty:
            for index, row in self.image_data.iterrows():
                image_path = row['Split_Image'] if pd.notna(row['Split_Image']) else row['Original_Image']
                is_left_image = row['Left_or_Right'] == 'Left' if pd.notna(row['Left_or_Right']) else True
                self.crop_to_largest_white_area(image_path, is_left_image)
            
            self.show_current_image()
            self.status = "changed"
            messagebox.showinfo("Crop Complete", "All images have been cropped to their largest white area.")    
    
    def crop_active_image(self):
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            # Determine if it's a left image based on the 'Left_or_Right' column
            is_left_image = current_image_row['Left_or_Right'] == 'Left' if pd.notna(current_image_row['Left_or_Right']) else True
            
            self.crop_to_largest_white_area(image_path, is_left_image)
            self.show_current_image()
            self.status = "changed"

    def auto_find_threshold(self, image_path):
        """Automatically find the optimal threshold for document detection"""
        # Read the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize variables for best threshold
        best_threshold = 0
        max_score = 0
        
        # Try different threshold values
        for threshold in range(0, 255, 5):  # Step of 5 for efficiency
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate metrics for this threshold
            white_pixels = np.sum(binary == 255)
            total_pixels = binary.size
            white_ratio = white_pixels / total_pixels
            
            # Find contours at this threshold
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape metrics
            if perimeter == 0:
                continue
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Score this threshold based on multiple criteria
            # - white_ratio should be reasonable (not too high or low)
            # - compactness should be close to 1 (for rectangular shapes)
            # - area should be significant
            white_score = 1.0 - abs(0.7 - white_ratio)  # Prefer ~70% white
            shape_score = compactness  # Higher for more rectangular shapes
            area_score = area / (image.shape[0] * image.shape[1])  # Normalized area
            
            # Combine scores
            score = white_score * shape_score * area_score
            
            if score > max_score:
                max_score = score
                best_threshold = threshold
        
        return best_threshold

    def auto_crop_image(self, event=None):
        # Clear cursor modes when auto cropping
        self.clear_all_modes()
        
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            ThresholdAdjuster(self, image_path)

    def auto_crop_all_images(self):
        """Apply auto-cropping to all images in the collection."""
        # Clear cursor modes when auto cropping all
        self.clear_all_modes()
        
        if not self.image_data.empty:
            # First, open threshold adjuster for the current image to get settings
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            current_image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            # Create a custom ThresholdAdjuster that will return values instead of applying them
            class SetupAdjuster(ThresholdAdjuster):
                def apply_crop(self):
                    self.return_values = {
                        'threshold': self.threshold_var.get(),
                        'margin': self.margin_var.get()
                    }
                    self.destroy()

            # Show the adjuster and wait for it to close
            adjuster = SetupAdjuster(self, current_image_path)
            self.wait_window(adjuster)
            
            # Check if values were set
            if not hasattr(adjuster, 'return_values'):
                return
            
            threshold = adjuster.return_values['threshold']
            margin = adjuster.return_values['margin']

            # Create progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Auto-cropping Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, 
                                    text="Processing images...",
                                    padding=10)
            progress_label.pack()
            
            progress_bar = ttk.Progressbar(progress_window, 
                                        length=200, 
                                        mode='determinate')
            progress_bar.pack(pady=20)
            
            total_images = len(self.image_data)
            progress_bar['maximum'] = total_images
            processed_count = tk.IntVar(value=0)

            def update_progress():
                progress_bar['value'] = processed_count.get()
                progress_label.config(text=f"Processing image {processed_count.get()} of {total_images}")
                progress_window.update()

            def process_image(image_path):
                try:
                    # Read the image
                    image = cv2.imread(image_path)
                    height, width = image.shape[:2]
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply threshold
                    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Get bounding rectangle with margin
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(width - x, w + 2 * margin)
                        h = min(height - y, h + 2 * margin)
                        
                        # Crop the image
                        cropped = image[y:y+h, x:x+w]
                        
                        # Save the cropped image with high quality
                        cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
                    
                    processed_count.set(processed_count.get() + 1)
                    self.after(10, update_progress)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")

            def process_all_images():
                try:
                    # Create a thread pool
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
                        # Get all image paths
                        image_paths = [
                            row['Split_Image'] if pd.notna(row['Split_Image']) else row['Original_Image']
                            for _, row in self.image_data.iterrows()
                        ]
                        
                        # Submit all tasks
                        futures = [executor.submit(process_image, path) for path in image_paths]
                        concurrent.futures.wait(futures)
                    
                    # Show completion message
                    progress_label.config(text="Auto-cropping completed!")
                    
                    # Close progress window and update display after delay
                    def cleanup():
                        progress_window.destroy()
                        self.show_current_image()
                        self.status = "changed"
                    
                    self.after(1000, cleanup)
                    
                except Exception as e:
                    messagebox.showerror("Error", 
                                    f"An error occurred while processing images: {str(e)}")
                    progress_window.destroy()

            # Start processing in a separate thread
            threading.Thread(target=process_all_images, daemon=True).start()
    
    def crop_to_largest_white_area(self, image_path, threshold=127, margin=10):
        try:
            # Read the image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle with margin
                x, y, w, h = cv2.boundingRect(largest_contour)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(width - x, w + 2 * margin)
                h = min(height - y, h + 2 * margin)
                
                # Crop the image
                cropped = image[y:y+h, x:x+w]
                
                # Save the cropped image with high quality
                cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
                
                # Force image reload if this is the current image
                current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
                current_image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
                
                if image_path == current_image_path:
                    self.show_current_image()
                
                return True
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during auto-crop: {str(e)}")
            return False

# Auto Straighten Functions

    def auto_straighten_image(self, event=None):
        """Automatically straighten the current image by detecting vertical edges."""
        # Clear cursor modes when auto straightening
        self.clear_all_modes()
        
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            angle = self.detect_straighten_angle(image_path)
            if angle is not None:
                # Apply the rotation
                self.rotate_image_by_angle(image_path, angle)
                messagebox.showinfo("Auto Straighten", f"Image straightened by {angle:.2f} degrees.")
            else:
                messagebox.showwarning("Auto Straighten", "Could not detect a clear vertical edge to straighten against.")

    def auto_straighten_all_images(self):
        """Apply auto straightening to all images in the collection."""
        # Clear cursor modes when auto straightening all
        self.clear_all_modes()
        
        if not self.image_data.empty:
            # Create progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Auto Straightening Progress")
            progress_window.geometry("350x150")
            progress_window.transient(self)
            progress_window.grab_set()

            # Progress label and bar
            progress_label = tk.Label(progress_window, text="Auto straightening images...")
            progress_label.pack(pady=10)

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=10, padx=20, fill=tk.X)

            status_label = tk.Label(progress_window, text="Starting...")
            status_label.pack(pady=5)

            # Cancel flag
            cancel_flag = {'cancelled': False}
            
            def cancel_processing():
                cancel_flag['cancelled'] = True
                progress_window.destroy()

            cancel_button = tk.Button(progress_window, text="Cancel", command=cancel_processing)
            cancel_button.pack(pady=5)

            def update_progress():
                progress_window.update()

            def process_image(image_path):
                if cancel_flag['cancelled']:
                    return False
                try:
                    angle = self.detect_straighten_angle(image_path)
                    if angle is not None and abs(angle) > 0.1:  # Only rotate if angle is significant
                        return self.rotate_image_by_angle(image_path, angle)
                    return True  # Consider it successful even if no rotation needed
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    return False

            def process_all_images():
                try:
                    total_images = len(self.image_data)
                    processed = 0
                    successful = 0
                    straightened = 0

                    for index, row in self.image_data.iterrows():
                        if cancel_flag['cancelled']:
                            break

                        image_path = row['Split_Image'] if pd.notna(row['Split_Image']) else row['Original_Image']
                        
                        # Update status
                        status_label.config(text=f"Processing: {os.path.basename(image_path)}")
                        update_progress()

                        # Detect angle first to count how many actually need straightening
                        angle = self.detect_straighten_angle(image_path)
                        
                        # Process the image
                        if process_image(image_path):
                            successful += 1
                            if angle is not None and abs(angle) > 0.1:
                                straightened += 1

                        processed += 1
                        progress_var.set((processed / total_images) * 100)
                        update_progress()

                    if not cancel_flag['cancelled']:
                        # Refresh current image display
                        def restore_view():
                            self.show_current_image()
                            self.status = "changed"
                            progress_window.destroy()
                            if successful == total_images:
                                messagebox.showinfo("Complete", f"Successfully processed all {successful} images.\n{straightened} images were straightened.")
                            else:
                                messagebox.showwarning("Partial Success", f"Processed {successful} out of {total_images} images.\n{straightened} images were straightened.")

                        self.after(100, restore_view)
                    else:
                        progress_window.destroy()

                except Exception as e:
                    messagebox.showerror("Error", 
                                    f"An error occurred while processing images: {str(e)}")
                    progress_window.destroy()

            # Start processing in a separate thread
            threading.Thread(target=process_all_images, daemon=True).start()

    def detect_straighten_angle(self, image_path):
        """Detect the angle needed to straighten the image by finding the most vertical edge."""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply mild Gaussian blur to reduce noise while preserving edges
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Try multiple Canny thresholds for better edge detection
            edge_candidates = []
            
            # Adaptive thresholds based on image statistics
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # For high contrast images (like white text on black), use lower thresholds
            if std_val > 50:  # High contrast
                canny_params = [(30, 90), (50, 150), (70, 210)]
            else:  # Lower contrast
                canny_params = [(50, 150), (80, 240), (100, 300)]
            
            # Try different Hough thresholds too
            hough_thresholds = [50, 75, 100, 150]
            
            all_vertical_lines = []
            
            for low, high in canny_params:
                edges = cv2.Canny(blurred, low, high, apertureSize=3)
                
                for threshold in hough_thresholds:
                    # Use Hough line transform to detect lines
                    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
                    
                    if lines is not None:
                        for line in lines:
                            rho, theta = line[0]
                            # Convert theta to degrees
                            angle_deg = np.degrees(theta)
                            
                            # Check if the line is close to vertical (90 degrees ± 30 degrees for better detection)
                            # Hough transform gives angles from 0 to 180, where 90 is vertical
                            deviation_from_vertical = min(abs(angle_deg - 90), abs(angle_deg - 90 + 180))
                            
                            if deviation_from_vertical < 30:  # More generous tolerance
                                # Calculate how much we need to rotate to make it perfectly vertical
                                if angle_deg > 90:
                                    rotation_needed = -(angle_deg - 90)  # Clockwise rotation
                                else:
                                    rotation_needed = 90 - angle_deg     # Counter-clockwise rotation
                                
                                # Store with line strength (inverse of deviation + rho factor)
                                line_strength = (30 - deviation_from_vertical) + abs(rho) / 1000
                                all_vertical_lines.append((deviation_from_vertical, rotation_needed, rho, line_strength))
            
            if not all_vertical_lines:
                print(f"No vertical lines detected in {os.path.basename(image_path)}")
                return None
            
            # Sort by deviation from vertical (smaller deviation = more vertical)
            all_vertical_lines.sort(key=lambda x: x[0])
            
            # Take the most vertical line
            best_deviation, best_rotation, best_rho, best_strength = all_vertical_lines[0]
            
            print(f"Detected angle for {os.path.basename(image_path)}: {best_rotation:.2f}° (deviation: {best_deviation:.1f}°)")
            
            # Only return significant angles (more than 0.1 degrees)
            if abs(best_rotation) > 0.1:
                return best_rotation
            else:
                print(f"Angle too small ({best_rotation:.2f}°), skipping rotation")
                return None
                
        except Exception as e:
            print(f"Error detecting straighten angle for {image_path}: {str(e)}")
            return None

    def rotate_image_by_angle(self, image_path, angle):
        """Rotate an image by a specific angle in degrees."""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Calculate the rotation matrix
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new image dimensions to fit the rotated image
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust the rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply the rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))  # White background
            
            # Save the rotated image with high quality
            cv2.imwrite(image_path, rotated, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            # Force image reload if this is the current image
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            current_image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            if image_path == current_image_path:
                self.show_current_image()
            
            return True
            
        except Exception as e:
            print(f"Error rotating image {image_path} by {angle} degrees: {str(e)}")
            return False
    
    def crop_grayscale_image(self, image_path, is_left_image):
        # This is the working grayscale version
        image = Image.open(image_path)
        
        # Convert to grayscale if it's not already
        if image.mode != 'L':
            image = image.convert('L')
        
        gray = np.array(image)
        
        # Apply thresholding
        threshold = 75
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add margin
            margin = 75
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.size[0] - x, w + 2 * margin)
            h = min(image.size[1] - y, h + 2 * margin)
            
            # Adjust based on left/right image
            if is_left_image:
                w = image.size[0] - x
            else:
                w = w + x
                x = 0
            
            # Crop and save
            original_image = Image.open(image_path)  # Open original image for cropping
            cropped_image = original_image.crop((x, y, x + w, y + h))
            cropped_image.save(image_path, 'JPEG', quality=95)

    def crop_color_image(self, image_path, is_left_image):
        # This is the working color version
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Separate vertical and horizontal lines
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    horizontal_lines.append(line[0])
                else:
                    vertical_lines.append(line[0])
            
            # Find document boundaries
            left_bound = image.shape[1]
            right_bound = 0
            top_bound = image.shape[0]
            bottom_bound = 0
            
            # Process vertical lines
            for line in vertical_lines:
                x1, y1, x2, y2 = line
                x_avg = (x1 + x2) // 2
                left_bound = min(left_bound, x_avg)
                right_bound = max(right_bound, x_avg)
            
            # Process horizontal lines
            for line in horizontal_lines:
                x1, y1, x2, y2 = line
                y_avg = (y1 + y2) // 2
                top_bound = min(top_bound, y_avg)
                bottom_bound = max(bottom_bound, y_avg)
            
            # Add margin
            margin = 20
            left_bound = max(0, left_bound - margin)
            right_bound = min(image.shape[1], right_bound + margin)
            top_bound = max(0, top_bound - margin)
            bottom_bound = min(image.shape[0], bottom_bound + margin)
            
            # Adjust bounds based on left/right image
            if is_left_image:
                right_bound = image.shape[1]
            else:
                left_bound = 0
            
            # Convert back to PIL and save
            image_pil = Image.open(image_path)
            cropped_image = image_pil.crop((left_bound, top_bound, right_bound, bottom_bound))
            cropped_image.save(image_path, 'JPEG', quality=95)
        else:
            # Fallback to threshold-based method if no lines detected
            threshold = 75
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add margin
                margin = 75
                x = max(0, x - margin)
                y = max(0, y - margin)
                
                # Get original image dimensions
                image_pil = Image.open(image_path)
                w = min(image_pil.size[0] - x, w + 2 * margin)
                h = min(image_pil.size[1] - y, h + 2 * margin)
                
                if is_left_image:
                    w = image_pil.size[0] - x
                else:
                    w = w + x
                    x = 0
                
                cropped_image = image_pil.crop((x, y, x + w, y + h))
                cropped_image.save(image_path, 'JPEG', quality=95)
    
    def activate_crop_tool(self, event=None):
        """Activate the crop tool"""
        # Clear all existing modes and bindings
        self.clear_all_modes()
        
        # Set up crop mode
        self.cropping = True
        self.special_cursor_active = False
        self.image_canvas.config(cursor="crosshair")
        
        # Set up crop-specific bindings
        self.image_canvas.bind("<ButtonPress-1>", self.start_crop)
        self.image_canvas.bind("<B1-Motion>", self.draw_crop)
        self.image_canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.bind("<Return>", self.apply_crop)
        self.bind("<Escape>", self.cancel_crop)

    def handle_mouse_release(self, event):
        """Handle mouse release event for cropping"""
        if self.cropping:
            self.crop_end = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
            if self.batch_process.get():
                self.apply_crop()  # Automatically apply crop if batch processing is enabled
            # Otherwise, wait for Enter key

    def start_crop(self, event):
        self.crop_start = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
        if self.crop_rect:
            self.image_canvas.delete(self.crop_rect)

    def draw_crop(self, event):
        if self.crop_start:
            x, y = self.crop_start
            if self.crop_rect:
                self.image_canvas.delete(self.crop_rect)
            self.crop_end = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
            self.crop_rect = self.image_canvas.create_rectangle(x, y, *self.crop_end, outline="red")

    def apply_crop(self, event=None):
        if self.crop_start and self.crop_end:
            try:
                # Get current image information
                current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
                image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
                
                # Load the image
                image = Image.open(image_path)
                
                # Convert canvas coordinates to image coordinates
                x1, y1 = self.crop_start
                x2, y2 = self.crop_end
                
                # Convert from canvas coordinates to image coordinates
                x1 = int(x1 / self.current_scale)
                y1 = int(y1 / self.current_scale)
                x2 = int(x2 / self.current_scale)
                y2 = int(y2 / self.current_scale)
                
                # Ensure coordinates are in the correct order
                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)
                
                # Crop the image
                cropped_image = image.crop((left, top, right, bottom))
                cropped_image.save(image_path, 'JPEG', quality=98, optimize=True)
                
                # Clear crop rectangle and reset crop variables
                if self.crop_rect:
                    self.image_canvas.delete(self.crop_rect)
                self.crop_rect = None
                self.crop_start = None
                self.crop_end = None
                
                # Update display
                self.show_current_image()
                
                # If batch processing is enabled, move to next image
                if self.batch_process.get():
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, self.activate_crop_tool)  # Reactivate crop tool for next image
                else:
                    self.clear_all_modes()
                
                self.status = "changed"
                
            except Exception as e:
                messagebox.showerror("Error", f"Error applying crop: {str(e)}")
                self.clear_all_modes()

    def end_crop(self, event):
        self.crop_end = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
        if self.batch_process.get():
            self.apply_crop(event)
            # After applying crop, move to next image

    def cancel_crop(self, event):
        self.clear_all_modes()

# Split Image Functions

    def split_all_images(self):
        if not self.image_data.empty:
            # Create progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Auto-splitting Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, 
                                    text="Processing images...",
                                    padding=10)
            progress_label.pack()
            
            progress_bar = ttk.Progressbar(progress_window, 
                                        length=200, 
                                        mode='determinate')
            progress_bar.pack(pady=20)
            
            # Count only unsplit images
            unsplit_images = self.image_data[self.image_data['Split_Image'].isna()]
            total_images = len(unsplit_images)
            progress_bar['maximum'] = total_images
            
            # Store the initial cursor line properties
            initial_cursor_orientation = self.cursor_orientation
            if self.cursor_orientation == 'vertical':
                initial_line_coords = self.image_canvas.coords(self.vertical_line)
            elif self.cursor_orientation == 'horizontal':
                initial_line_coords = self.image_canvas.coords(self.horizontal_line)
            elif self.cursor_orientation == 'angled':
                initial_line_coords = self.image_canvas.coords(self.cursor_line)
                initial_angle = self.cursor_angle
            else:
                messagebox.showerror("Error", "Please position the split line first")
                progress_window.destroy()
                return

            def process_images():
                try:
                    processed_count = 0
                    current_idx = self.current_image_index
                    
                    while current_idx < len(self.image_data):
                        # Get current row
                        current_row = self.image_data.iloc[current_idx]
                        
                        # Skip if image is already split
                        if pd.notna(current_row['Split_Image']):
                            current_idx += 1
                            continue
                        
                        # Update progress
                        processed_count += 1
                        progress_bar['value'] = processed_count
                        progress_label.config(text=f"Processing image {processed_count} of {total_images}")
                        progress_window.update()
                        
                        # Navigate to the image
                        self.current_image_index = current_idx
                        self.show_current_image()
                        
                        # Restore the original cursor line
                        self.cursor_orientation = initial_cursor_orientation
                        if initial_cursor_orientation == 'vertical':
                            self.vertical_line = self.image_canvas.create_line(
                                *initial_line_coords, fill="red", width=2)
                        elif initial_cursor_orientation == 'horizontal':
                            self.horizontal_line = self.image_canvas.create_line(
                                *initial_line_coords, fill="red", width=2)
                        elif initial_cursor_orientation == 'angled':
                            self.cursor_angle = initial_angle
                            self.cursor_line = self.image_canvas.create_line(
                                *initial_line_coords, fill="red", width=2)
                        
                        # Split the image
                        self.split_image_manually()
                        
                        # Move to next unsplit image (skip the newly created right image)
                        current_idx += 2
                    
                    # Show completion message
                    progress_label.config(text="Auto-splitting completed!")
                    
                    # Return to original image after short delay
                    def restore_view():
                        self.show_current_image()
                        progress_window.destroy()
                    
                    self.after(1000, restore_view)
                    
                except Exception as e:
                    messagebox.showerror("Error", 
                                    f"An error occurred while processing images: {str(e)}")
                    progress_window.destroy()
            
            # Start processing in a separate thread
            threading.Thread(target=process_images, daemon=True).start()
    
    def split_straight_cursor(self, image, width, height):
        if self.vertical_line:
            try:
                # Get the x-coordinate of the vertical line
                split_x = int(self.image_canvas.coords(self.vertical_line)[0] / self.current_scale)
                
                # Ensure split_x is within image bounds
                split_x = max(0, min(split_x, width))
                
                # Create left and right images
                left_image = image.crop((0, 0, split_x, height))
                right_image = image.crop((split_x, 0, width, height))
                
                
                return left_image, right_image
            except Exception as e:
                print(f"Error in split_straight_cursor: {str(e)}")
                return None, None
        return None, None

    def split_horizontal_cursor(self, image, width, height):
        if self.horizontal_line:
            # Get the y-coordinate of the horizontal line
            split_y = int(self.image_canvas.coords(self.horizontal_line)[1] / self.current_scale)
            
            # Ensure split_y is within image bounds
            split_y = max(0, min(split_y, height))
            
            # Create top and bottom images
            top_image = image.crop((0, 0, width, split_y))
            bottom_image = image.crop((0, split_y, width, height))
            
            return top_image, bottom_image
        return None, None

    def split_image_manually(self):
        if not self.image_data.empty:
            try:
                # Get current image information
                current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
                current_image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
                
                image = Image.open(current_image_path).convert("RGB")
                width, height = image.size
                
                # Check if cursor line exists and is active
                if not self.special_cursor_active:
                    messagebox.showerror("Error", "Please activate the split tool first (Ctrl+V for vertical or Ctrl+H for horizontal)")
                    return
                
                # Determine split type based on cursor state
                if self.cursor_orientation == 'angled' and self.cursor_line:
                    left_image, right_image = self.angled_cursor_split(image, width, height)
                    split_type = 'angled'
                elif self.cursor_orientation == 'vertical' and self.vertical_line:
                    left_image, right_image = self.split_straight_cursor(image, width, height)
                    split_type = 'vertical'
                elif self.cursor_orientation == 'horizontal' and self.horizontal_line:
                    left_image, right_image = self.split_horizontal_cursor(image, width, height)
                    split_type = 'horizontal'
                else:
                    messagebox.showerror("Error", "Please position the split line first")
                    return
                
                if left_image is None or right_image is None:
                    messagebox.showerror("Error", "Failed to create split images")
                    return
                    
                # Process the split images
                self.process_split_image(current_image_path, current_image_row, left_image, right_image, split_type)
                
                # Clear cursor lines
                self.clear_cursor_lines()
                
                # Update display
                self.show_current_image()
                self.status = "changed"
                
            except Exception as e:
                messagebox.showerror("Error", f"Error splitting image: {str(e)}")
    
    def revert_to_original(self):
        """Revert current image to its original state"""
        # Clear cursor modes when reverting
        self.clear_all_modes()
        
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            
            # Get paths
            original_image_path = current_image_row['Original_Image']
            
            # Get the original filename (without path)
            original_filename = os.path.basename(original_image_path)
            
            # Path to the backup of the truly original file
            backup_original_path = os.path.join(self.originals_folder, original_filename)
            
            # Try to find the original backup - use several methods if direct path fails
            backup_found = False
            original_backup_file = None
            
            # Method 1: Direct path match
            if os.path.exists(backup_original_path):
                backup_found = True
                original_backup_file = backup_original_path
            else:
                # Method 2: Try to find by stripping formatting and sequence numbers
                try:
                    # Strip sequence numbers and formatting
                    base_filename_pattern = ''.join(c for c in original_filename if not (c.isdigit() or c in '_.'))
                    
                    if base_filename_pattern and os.path.exists(self.originals_folder):
                        # Search for files with similar names
                        for backup_file in os.listdir(self.originals_folder):
                            if backup_file.lower().endswith(('.jpg', '.jpeg')):
                                # Check if this backup file could match our original
                                if base_filename_pattern in backup_file:
                                    backup_found = True
                                    original_backup_file = os.path.join(self.originals_folder, backup_file)
                                    break
                except Exception as e:
                    print(f"Error in backup file search: {str(e)}")
                
                # Method 3: If there's only one file in the originals folder, use that
                if not backup_found and os.path.exists(self.originals_folder):
                    backup_files = [f for f in os.listdir(self.originals_folder) 
                                 if f.lower().endswith(('.jpg', '.jpeg'))]
                    if len(backup_files) == 1:
                        backup_found = True
                        original_backup_file = os.path.join(self.originals_folder, backup_files[0])
            
            # Check if we found a backup by any method
            if backup_found and original_backup_file:
                # Find all rows with this original image (both left and right splits)
                related_rows = self.image_data[self.image_data['Original_Image'] == original_image_path]
                current_index = current_image_row.name
                
                # Delete all split images related to this original
                for _, row in related_rows.iterrows():
                    if pd.notna(row['Split_Image']):
                        split_path = row['Split_Image']
                        if os.path.exists(split_path):
                            try:
                                os.remove(split_path)
                            except Exception as e:
                                print(f"Error removing split image {split_path}: {str(e)}")
                
                # Remove all related rows from the DataFrame
                self.image_data = self.image_data[self.image_data['Original_Image'] != original_image_path]
                
                # Now restore the truly original image from our backup
                # Copy from backup to restore original image
                try:
                    shutil.copy2(original_backup_file, original_image_path)
                    print(f"Restored original from: {original_backup_file} to {original_image_path}")
                except Exception as e:
                    print(f"Error copying backup file: {str(e)}")
                    messagebox.showerror("Error", f"Failed to restore original image: {str(e)}")
                    return
                
                # Add a new row for the original image
                original_row = pd.DataFrame({
                    'Image_Index': [current_index + 1],
                    'Original_Image': [original_image_path],
                    'Split_Image': [None],
                    'Left_or_Right': [None]
                })
                
                # Insert the original image at the appropriate position
                self.image_data = pd.concat([
                    self.image_data.iloc[:current_index] if current_index > 0 else pd.DataFrame(),
                    original_row,
                    self.image_data.iloc[current_index:] if current_index < len(self.image_data) else pd.DataFrame()
                ], ignore_index=True)
                
                # Reset the index of the DataFrame
                self.image_data.reset_index(drop=True, inplace=True)
                
                # Update the Image_Index column
                self.image_data['Image_Index'] = self.image_data.index + 1
                
                # Move to the reverted original image
                self.current_image_index = current_index
                
                # Force update of display
                self.after(50, self.show_current_image)
                self.status = "changed"
            else:
                print(f"Could not find original backup for: {original_filename}")
                print(f"Original path: {original_image_path}")
                print(f"Looked in: {self.originals_folder}")
                if os.path.exists(self.originals_folder):
                    print(f"Available backups: {os.listdir(self.originals_folder)}")
                messagebox.showwarning("Revert Failed", "Original backup image not found.")
                return

    def revert_all_images(self):
        """Revert all images to their original state"""
        # Clear cursor modes when reverting all
        self.clear_all_modes()
        
        if not self.image_data.empty:
            # Create a progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Reverting Images")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, 
                                    text="Reverting images...",
                                    padding=10)
            progress_label.pack()
            
            progress_bar = ttk.Progressbar(progress_window, 
                                        length=200, 
                                        mode='determinate')
            progress_bar.pack(pady=20)
            
            # Get all unique original images
            original_images = self.image_data['Original_Image'].unique()
            total_images = len(original_images)
            progress_bar['maximum'] = total_images
            
            def process_images():
                try:
                    # Create a new DataFrame for the reverted images
                    new_image_data = pd.DataFrame(columns=self.image_data.columns)
                    errors = []
                    
                    for i, original_image_path in enumerate(original_images):
                        # Update progress
                        progress_bar['value'] = i + 1
                        progress_label.config(text=f"Reverting image {i + 1} of {total_images}")
                        progress_window.update()
                            
                        # Get the original filename (without path)
                        original_filename = os.path.basename(original_image_path)
                        
                        # Path to the backup of the truly original file
                        backup_original_path = os.path.join(self.originals_folder, original_filename)
                        
                        # Try to find the original backup file with multiple methods
                        backup_found = False
                        original_backup_file = None
                        
                        # Method 1: Direct path match
                        if os.path.exists(backup_original_path):
                            backup_found = True
                            original_backup_file = backup_original_path
                        else:
                            # Method 2: Try to find by stripping formatting and sequence numbers
                            try:
                                # Strip sequence numbers and formatting
                                base_filename_pattern = ''.join(c for c in original_filename if not (c.isdigit() or c in '_.'))
                                
                                if base_filename_pattern and os.path.exists(self.originals_folder):
                                    # Search for files with similar names
                                    for backup_file in os.listdir(self.originals_folder):
                                        if backup_file.lower().endswith(('.jpg', '.jpeg')):
                                            # Check if this backup file could match our original
                                            if base_filename_pattern in backup_file:
                                                backup_found = True
                                                original_backup_file = os.path.join(self.originals_folder, backup_file)
                                                break
                            except Exception as e:
                                print(f"Error in backup file search for {original_filename}: {str(e)}")
                            
                            # Method 3: If there's only one file in the originals folder and we're processing only one image, use that
                            if not backup_found and os.path.exists(self.originals_folder) and len(original_images) == 1:
                                backup_files = [f for f in os.listdir(self.originals_folder) 
                                    if f.lower().endswith(('.jpg', '.jpeg'))]
                                if len(backup_files) == 1:
                                    backup_found = True
                                    original_backup_file = os.path.join(self.originals_folder, backup_files[0])
                        
                        if backup_found and original_backup_file:
                            # Get all rows with this original image
                            related_rows = self.image_data[self.image_data['Original_Image'] == original_image_path]
                            
                            # Delete all split image files
                            for _, row in related_rows.iterrows():
                                if pd.notna(row['Split_Image']):
                                    split_image_path = row['Split_Image']
                                    if os.path.exists(split_image_path):
                                        try:
                                            os.remove(split_image_path)
                                        except Exception as e:
                                            print(f"Error removing split image {split_image_path}: {str(e)}")
                            
                            # Copy from backup to restore original image
                            try:
                                shutil.copy2(original_backup_file, original_image_path)
                                print(f"Restored original from: {original_backup_file} to {original_image_path}")
                            except Exception as e:
                                print(f"Error copying backup file for {original_filename}: {str(e)}")
                                errors.append(f"Failed to restore {original_filename}: {str(e)}")
                                continue
                            
                            # Add back original image row
                            original_row = pd.DataFrame({
                                'Image_Index': [len(new_image_data) + 1],
                                'Original_Image': [original_image_path],
                                'Split_Image': [None],
                                'Left_or_Right': [None]
                            })
                            
                            new_image_data = pd.concat([new_image_data, original_row], ignore_index=True)
                        else:
                            # Log the error
                            error_msg = f"Could not find original backup for: {original_filename}"
                            print(error_msg)
                            print(f"Original path: {original_image_path}")
                            print(f"Looked in: {self.originals_folder}")
                            if os.path.exists(self.originals_folder):
                                print(f"Available backups: {os.listdir(self.originals_folder)}")
                            
                            errors.append(error_msg)
                            
                            # If no backup exists, keep just one instance of the original
                            if len(related_rows) > 0:
                                row_data = pd.DataFrame({
                                    'Image_Index': [len(new_image_data) + 1],
                                    'Original_Image': [original_image_path],
                                    'Split_Image': [None],
                                    'Left_or_Right': [None]
                                })
                                
                                new_image_data = pd.concat([new_image_data, row_data], ignore_index=True)
                    
                    # Replace the current image_data with the new one
                    self.image_data = new_image_data
                    
                    # Reset index and update Image_Index
                    self.image_data.reset_index(drop=True, inplace=True)
                    self.image_data['Image_Index'] = self.image_data.index + 1
                    
                    # Reset current image index to beginning
                    self.current_image_index = 0
                    
                    # Show first image with slight delay to ensure UI is updated
                    self.after(100, self.show_current_image)
                    
                    # Show completion message
                    if errors:
                        progress_label.config(text=f"Reverted with {len(errors)} errors")
                    else:
                        progress_label.config(text="All images reverted!")
                    
                    def close_progress():
                        progress_window.destroy()
                        self.status = "changed"
                        # Force another refresh of display
                        self.after(50, self.show_current_image)
                        
                        # Show any errors in a separate dialog
                        if errors:
                            error_message = "\n".join(errors[:10])
                            if len(errors) > 10:
                                error_message += f"\n...and {len(errors) - 10} more errors"
                            messagebox.showwarning("Revert Warnings", 
                                               f"Some images could not be fully reverted:\n\n{error_message}")
                    
                    self.after(1000, close_progress)
                    
                except Exception as e:
                    print(f"Error in revert_all_images: {str(e)}")
                    messagebox.showerror("Error", f"An error occurred while reverting images: {str(e)}")
                    progress_window.destroy()
            
            # Start processing in a separate thread
            threading.Thread(target=process_images, daemon=True).start()

# Angled Split Functions

    def rotate_cursor(self, direction):
        """
        Rotate the cursor starting from its current orientation
        direction: -1 for counter-clockwise, 1 for clockwise
        """
        if not self.special_cursor_active:
            return
        
        # Set initial angle based on current orientation if we're just starting rotation
        if self.cursor_orientation in ['vertical', 'horizontal']:
            # Start from 90 degrees if vertical, 0 degrees if horizontal
            self.cursor_angle = 90 if self.cursor_orientation == 'vertical' else 0
            self.cursor_orientation = 'angled'
        
        # Rotate by 1 degree in the specified direction
        self.cursor_angle = (self.cursor_angle + direction) % 360
        
        # Force cursor update with current mouse position
        mock_event = type('MockEvent', (), {
            'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
            'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
        })
        self.update_cursor_line(mock_event)

    def angled_cursor_split(self, image, width, height):
        """Split image along the angled cursor line"""
        if self.cursor_line is None:
            return None, None
            
        try:
            # Get cursor line coordinates
            x1, y1, x2, y2 = self.image_canvas.coords(self.cursor_line)
            
            # Convert canvas coordinates to image coordinates
            x1 = int(x1 / self.current_scale)
            y1 = int(y1 / self.current_scale)
            x2 = int(x2 / self.current_scale)
            y2 = int(y2 / self.current_scale)
            
            # Create mask images
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            # Determine how to fill the mask based on original orientation
            if self.cursor_orientation == 'horizontal' or self.cursor_angle < 45 or self.cursor_angle > 315:
                # For horizontal-based splits, fill above the line
                points = [(0, 0), (width, 0), (x2, y2), (x1, y1)]
            else:
                # For vertical-based splits, fill left of the line
                points = [(0, 0), (x1, y1), (x2, y2), (0, height)]
                
            # Draw the polygon to create the mask
            draw.polygon(points, fill=255)
            
            # Create left and right images
            left_image = Image.new('RGB', (width, height), (0, 0, 0))
            right_image = Image.new('RGB', (width, height), (0, 0, 0))
            
            # Copy the appropriate parts of the original image
            left_image.paste(image, mask=mask)
            right_image.paste(image, mask=ImageChops.invert(mask))
            
            # Crop images to content
            left_bbox = left_image.convert('L').getbbox()
            right_bbox = right_image.convert('L').getbbox()
            
            if left_bbox:
                left_image = left_image.crop(left_bbox)
            if right_bbox:
                right_image = right_image.crop(right_bbox)
                
            return left_image, right_image
            
        except Exception as e:
            print(f"Error in angled_cursor_split: {str(e)}")
            return None, None

# Image Rotation Functions

    def incremental_rotate(self):
        # Only clear cursor modes when rotating if NOT in batch processing mode
        if not self.batch_process.get():
            self.clear_all_modes()
            
        angle = simpledialog.askfloat("Rotate Image", "Enter rotation angle (positive for clockwise, negative for counter-clockwise):", initialvalue=0)
        if angle is not None:
            self.rotate_image(angle)

    def rotate_image(self, angle):
        # Only clear cursor modes when rotating if NOT in batch processing mode
        if hasattr(self, 'special_cursor_active') and self.special_cursor_active and not self.batch_process.get():
            self.clear_all_modes()
            
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            image_path = current_image_row['Split_Image'] if pd.notna(current_image_row['Split_Image']) else current_image_row['Original_Image']
            
            image = Image.open(image_path)
            rotated_image = image.rotate(-angle, expand=True, resample=Image.BICUBIC)  # Note the negative angle here
            rotated_image.save(image_path, 'JPEG', quality=98, optimize=True)
            
            self.show_current_image()
        
        self.status = "changed"
        self.show_current_image()
        
    def rotate_all_images(self, angle):
        # Clear cursor modes when rotating all images
        self.clear_all_modes()
        
        if not self.image_data.empty:
            for _, row in self.image_data.iterrows():
                image_path = row['Split_Image'] if pd.notna(row['Split_Image']) else row['Original_Image']
                
                image = Image.open(image_path)
                rotated_image = image.rotate(angle, expand=True)
                rotated_image.save(image_path, 'JPEG', quality=98, optimize=True)
            
            self.show_current_image()
        
        self.status = "changed"
        self.show_current_image()

# Image Navigation Functions

    def navigate_images(self, direction):
        # Only clear cursor modes when navigating if NOT in batch processing mode
        if hasattr(self, 'special_cursor_active') and self.special_cursor_active and not self.batch_process.get():
            self.clear_all_modes()
            
        # Get the total number of rows in the df
        total_images = len(self.image_data) - 1
        if direction == -2: # Go to the first image
            self.current_image_index = 0
        elif direction == -1: # Go to the previous image
            self.current_image_index = max(0, self.current_image_index - 1)
        elif direction == 1: # Go to the next image
            self.current_image_index = min(total_images, self.current_image_index + 1)
        elif direction == 2: # Go to the last image
            self.current_image_index = total_images
        self.show_current_image()
        self.ensure_cursor_bindings()

    def show_current_image(self):
        if self.current_image_index >= len(self.image_data):
            return
            
        current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
        
        # Determine which image path to use
        if pd.notna(current_image_row['Split_Image']):
            current_image_path = current_image_row['Split_Image']
        else:
            current_image_path = current_image_row['Original_Image']
            
        try:
            # Load and process the image
            image = Image.open(current_image_path)
            self.original_image = image  # Store the original image

            # Calculate scaling factors
            image_width, image_height = image.size
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            scale_x = canvas_width / image_width
            scale_y = canvas_height / image_height
            self.current_scale = min(scale_x, scale_y)

            # Resize the image
            new_width = int(image_width * self.current_scale)
            new_height = int(image_height * self.current_scale)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            # Display the image
            photo = ImageTk.PhotoImage(resized_image)
            self.image_canvas.delete("all")
            self.image_canvas.image = photo
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

            # If special cursor is active, redraw it
            if self.special_cursor_active:
                # Force cursor update with current mouse position
                mock_event = type('MockEvent', (), {
                    'x': self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx(),
                    'y': self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
                })
                self.update_cursor_line(mock_event)
                
        except FileNotFoundError:
            messagebox.showerror("Error", "Image file not found.")  

    def delete_current_image(self):
        # Clear cursor modes when deleting
        self.clear_all_modes()
        
        if not self.image_data.empty:
            current_image_row = self.image_data[self.image_data['Image_Index'] == self.current_image_index + 1].iloc[0]
            current_image_path = current_image_row['Original_Image']
            split_image_path = current_image_row['Split_Image']
            # Show confirmation dialog before deleting the image
            confirm = messagebox.askyesno("Delete Image", "Are you sure you want to delete the current image?")
            if confirm:
                if pd.notna(split_image_path):
                    # Delete the split image file
                    if os.path.exists(split_image_path):
                        os.remove(split_image_path)
                    # Update the corresponding left or right split image as the original image
                    left_or_right = current_image_row['Left_or_Right']
                    if left_or_right == 'Left':
                        right_image_path = os.path.splitext(split_image_path)[0][:-1] + '2.jpg'
                        right_image_row = self.image_data[self.image_data['Split_Image'] == right_image_path]
                        if not right_image_row.empty:
                            self.image_data.at[right_image_row.index[0], 'Original_Image'] = right_image_path
                            self.image_data.at[right_image_row.index[0], 'Split_Image'] = np.nan
                            self.image_data.at[right_image_row.index[0], 'Left_or_Right'] = np.nan
                    elif left_or_right == 'Right':
                        left_image_path = os.path.splitext(split_image_path)[0][:-1] + '1.jpg'
                        left_image_row = self.image_data[self.image_data['Split_Image'] == left_image_path]
                        if not left_image_row.empty:
                            self.image_data.at[left_image_row.index[0], 'Original_Image'] = left_image_path
                            self.image_data.at[left_image_row.index[0], 'Split_Image'] = np.nan
                            self.image_data.at[left_image_row.index[0], 'Left_or_Right'] = np.nan
                else:
                    # Delete the original image file
                    if os.path.exists(current_image_path):
                        os.remove(current_image_path)
                # Remove the current image row from the DataFrame
                self.image_data.drop(current_image_row.name, inplace=True)
                # Reset the index of the DataFrame
                self.image_data.reset_index(drop=True, inplace=True)
                # Update the Image_Index column
                self.image_data['Image_Index'] = self.image_data.index + 1
                # Update the current_image_index
                if self.current_image_index >= len(self.image_data):
                    self.current_image_index = max(0, len(self.image_data) - 1)
                self.show_current_image()
        
        self.status = "changed"
        self.show_current_image()
        
    def save_split_images(self, only_current=False):
        # Clear cursor modes when saving
        self.clear_all_modes()
        
        confirm = messagebox.askyesno("Save Images", "Are you sure you want to save the current project? This will finalize the images and cannot be undone.")
        if not confirm:
            return
        else:
            self.commit_changes(only_current=only_current)
    
    def commit_changes(self, only_current=False):
        if not self.folder_path or self.image_data.empty:
            return False

        try:
            # Set up and clean the pass_images directory
            pass_images_dir = self.ensure_pass_images_dir()

            # Determine which data to save
            if only_current:
                # Save only the current image's rows
                if self.image_data.empty:
                    return False
                current_idx = self.current_image_index
                # Get all rows with the same Original_Image as the current row
                current_row = self.image_data.iloc[current_idx]
                original_image = current_row['Original_Image']
                data_to_save = self.image_data[self.image_data['Original_Image'] == original_image].copy()
                # Reset index for naming
                data_to_save = data_to_save.reset_index(drop=True)
            else:
                # Save all images (default behavior)
                data_to_save = self.image_data.sort_values('Image_Index').reset_index(drop=True)

            # Process and save images sequentially
            for i, row in data_to_save.iterrows():
                try:
                    # Create consistent naming scheme - use sequential numbers
                    new_name = f"{i+1:04d}.jpg"
                    new_path = os.path.join(pass_images_dir, new_name)
                    # Determine source image - prefer Split_Image over Original_Image
                    source_path = row['Split_Image'] if pd.notna(row['Split_Image']) else row['Original_Image']
                    
                    # Validate source path exists
                    if not source_path or not os.path.exists(source_path):
                        print(f"Error: Source image not found at {source_path}")
                        continue
                    
                    # Copy image directly to preserve quality (avoid double compression)
                    shutil.copy2(source_path, new_path)
                    
                    # Keep track of the mapping between old and new paths
                    row['Final_Path'] = new_path
                except Exception as e:
                    print(f"Error processing image {i+1}: {e}")
                    continue

            self.status = "saved"
            self.destroy()
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images: {str(e)}")
            print(f"Error in commit_changes: {e}")
            return False

    def run(self):
        self.mainloop()

    def set_temp_dir(self):
        if self.folder_path:
            # Store the original directory path
            self.original_directory = self.folder_path
            
            # Check if the original directory exists
            if not os.path.exists(self.folder_path):
                print(f"Warning: Source directory does not exist: {self.folder_path}")
                # Create the directory if it doesn't exist
                try:
                    os.makedirs(self.folder_path, exist_ok=True)
                    print(f"Created directory: {self.folder_path}")
                except Exception as e:
                    print(f"Error creating directory {self.folder_path}: {e}")
            
            # Set up our own internal temp directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_folder = os.path.join(current_script_dir, "temp")
            
            # Create temp directory if it doesn't exist
            os.makedirs(self.temp_folder, exist_ok=True)
            
            # Set up original backups directory
            self.originals_folder = os.path.join(current_script_dir, "originals_backup")
            os.makedirs(self.originals_folder, exist_ok=True)
            
            # Clear existing files in our temp directory
            for item in os.listdir(self.temp_folder):
                item_path = os.path.join(self.temp_folder, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            # Clear existing files in our originals backup directory
            for item in os.listdir(self.originals_folder):
                item_path = os.path.join(self.originals_folder, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            # Copy images to temp directory if source directory has any files
            if os.path.exists(self.folder_path):
                try:
                    files_copied = 0
                    for file in os.listdir(self.folder_path):
                        lower_file = file.lower()
                        if lower_file.endswith((".jpg", ".jpeg")):
                            src = os.path.join(self.folder_path, file)
                            # Copy to temp (working directory)
                            dst = os.path.join(self.temp_folder, file)
                            shutil.copy2(src, dst)
                            # Also copy to originals backup
                            orig_dst = os.path.join(self.originals_folder, file)
                            shutil.copy2(src, orig_dst)
                            files_copied += 1
                        elif lower_file.endswith((".png", ".tif", ".tiff", ".jp2")):
                            # Convert non-JPEG formats to JPEG
                            src = os.path.join(self.folder_path, file)
                            try:
                                img = Image.open(src)
                                # Handle transparency by adding white background
                                if img.mode in ('RGBA', 'LA'):
                                    background = Image.new('RGB', img.size, (255, 255, 255))
                                    alpha_mask = img.split()[-1] if len(img.split()) > 3 else None
                                    background.paste(img, mask=alpha_mask)
                                    img = background
                                # Convert to RGB if not already
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')

                                # Save as JPEG with original base name
                                base_name = os.path.splitext(file)[0]
                                jpeg_name = base_name + '.jpg'
                                dst = os.path.join(self.temp_folder, jpeg_name)
                                orig_dst = os.path.join(self.originals_folder, jpeg_name)

                                img.save(dst, 'JPEG', quality=95)
                                img.save(orig_dst, 'JPEG', quality=95)
                                files_copied += 1
                            except Exception as convert_error:
                                print(f"Error converting {file} to JPEG: {convert_error}")

                    # If we didn't find any images, print a warning
                    if files_copied == 0:
                        print(f"Warning: No image files found in {self.folder_path}")
                except Exception as e:
                    print(f"Error copying files from {self.folder_path}: {e}")
            
            # Update folder_path to our internal temp
            self.folder_path = self.temp_folder

    def on_closing(self):
        if self.status != "no_changes":
            response = messagebox.askyesnocancel("Save Changes", "Do you want to save your changes before closing?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                # If only one image in the temp folder, treat as single-image edit
                only_current = False
                if hasattr(self, 'image_data') and len(self.image_data['Original_Image'].unique()) == 1:
                    only_current = True
                if self.commit_changes(only_current=only_current):
                    self.status = "saved"
                else:
                    # If saving failed and user wants to try again
                    if messagebox.askyesno("Save Failed", "Failed to save changes. Try again?"):
                        return  # Stay open for user to try again
            else:  # No
                self.status = "discarded"
                # Clean up the pass_images directory if we're discarding changes
                try:
                    self.ensure_pass_images_dir()  # This cleans up the directory too
                except Exception as e:
                    print(f"Error cleaning up pass_images directory: {e}")
        
        # Clean up temp folder regardless of save status
        try:
            if hasattr(self, 'temp_folder') and self.temp_folder and os.path.exists(self.temp_folder):
                # List all files before removing to check if any are locked
                for item in os.listdir(self.temp_folder):
                    try:
                        item_path = os.path.join(self.temp_folder, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                    except Exception as item_err:
                        print(f"Could not remove {item_path}: {item_err}")
                
                # Now try to remove the directory itself
                try:
                    shutil.rmtree(self.temp_folder, ignore_errors=True)
                except Exception as dir_err:
                    print(f"Could not completely remove temp folder: {dir_err}")
                    # If full removal fails, at least make sure we've removed content
                    for item in os.listdir(self.temp_folder):
                        try:
                            item_path = os.path.join(self.temp_folder, item)
                            if os.path.isfile(item_path):
                                os.chmod(item_path, 0o777)  # Try to ensure we have permission
                                os.remove(item_path)
                        except:
                            pass
                            
            # Clean up originals backup folder
            if hasattr(self, 'originals_folder') and self.originals_folder and os.path.exists(self.originals_folder):
                try:
                    # Remove all files in the originals backup directory
                    for item in os.listdir(self.originals_folder):
                        item_path = os.path.join(self.originals_folder, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                    
                    # Try to remove the directory itself
                    shutil.rmtree(self.originals_folder, ignore_errors=True)
                except Exception as e:
                    print(f"Error cleaning up originals backup folder: {e}")
        
        except Exception as e:
            print(f"Error during temp folder cleanup: {e}")
        
        self.destroy()

    def ensure_pass_images_dir(self):
        """Ensure the pass_images directory exists and is empty"""
        try:
            # Set up the pass_images directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            pass_images_dir = os.path.join(current_script_dir, "pass_images")
            
            # Create the pass_images directory if it doesn't exist
            os.makedirs(pass_images_dir, exist_ok=True)

            # Clear existing files in pass_images directory
            for item in os.listdir(pass_images_dir):
                item_path = os.path.join(pass_images_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                except Exception as e:
                    print(f"Error deleting file {item_path}: {e}")
                    
            return pass_images_dir
        except Exception as e:
            print(f"Error ensuring pass_images directory: {e}")
            raise

if __name__ == "__main__":
    app = ImageSplitter(".")
    app.run()

    

