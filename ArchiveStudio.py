import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import pandas as pd
import fitz, re, os, shutil, asyncio, difflib, ast
from PIL import Image, ImageTk, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import traceback

# Import natural_sort_key from DataOperations to use in edit_all_images
from util.DataOperations import DataOperations, natural_sort_key

# Import Local Scripts
from util.subs.ImageSplitter import ImageSplitter
from util.FindReplace import FindReplace
from util.APIHandler import APIHandler
from util.ProgressBar import ProgressBar
from util.SettingsWindow import SettingsWindow
from util.Settings import Settings
from util.ImageHandler import ImageHandler
from util.ProjectIO import ProjectIO
from util.ExportFunctions import ExportManager
from util.AdvancedDiffHighlighting import highlight_text_differences
from util.AIFunctions import AIFunctionsHandler
from util.ErrorLogger import log_error
from util.NamesAndPlaces import NamesAndPlacesHandler
from util.Highlights import HighlightHandler # <--- Added Import

class App(TkinterDnD.Tk):

# Basic Setup

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Archive Studio 1.2.2 macOS") # Set the window title
        self.link_nav = 0
        self.geometry("1200x800")

        # State Variables
        self.current_image_path_list = None # Stores the list of paths for the current document
        self.current_doc_page_index = 0    # Index of the image displayed within the list

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        if os.name == 'nt':  # For Windows use the .ico file
            try:
                self.iconbitmap("util/pb.ico")
            except:
                pass  # If icon file is not found, use default icon

        # Flags, Toggles, and Variables
        self.save_toggle = False
        self.find_replace_toggle = False
        self.original_image = None
        self.photo_image = None
        self.pagination_added = False
        self.highlight_names_var = tk.BooleanVar()
        self.highlight_places_var = tk.BooleanVar()
        self.highlight_changes_var = tk.BooleanVar()
        self.highlight_errors_var = tk.BooleanVar()
        self.skip_completed_pages = tk.BooleanVar(value=True)  # Default to skipping completed pages
        self.relevance_var = tk.StringVar() # Added for relevance dropdown
        self.text_font_size = 20  # Default font size for text display

        # Initialize settings early
        self.settings = Settings()
        
        # Initialize data operations early - BEFORE initializing main_df
        self.data_operations = DataOperations(self)
        # Initialize main_df using DataOperations
        self.data_operations.initialize_main_df()
        
        # Control visibility of bottom bar elements
        self.show_relevance = tk.BooleanVar(value=False)
        self.show_page_nav = tk.BooleanVar(value=False)
        # self.show_relevance_nav = tk.BooleanVar(value=True) # REMOVED - Visibility tied to show_relevance

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Top frame
        self.grid_rowconfigure(1, weight=1)  # Main frame
        self.grid_rowconfigure(2, weight=0)  # Bottom Bar frame

        # Modify the top frame configuration
        self.top_frame = tk.Frame(self)
        self.top_frame.grid(row=0, column=0, sticky="nsew")
        self.top_frame.grid_columnconfigure(1, weight=1)  # Make middle column expandable

        # Create groups in the top frame
        left_group = tk.Frame(self.top_frame)
        left_group.grid(row=0, column=0, sticky="w", padx=5)

        middle_group = tk.Frame(self.top_frame)
        middle_group.grid(row=0, column=1, sticky="ew", padx=5)

        right_group = tk.Frame(self.top_frame)
        right_group.grid(row=0, column=2, sticky="e", padx=(0, 5))

        # Left group elements (text type and toggle)
        text_label = tk.Label(left_group, text="Displayed Text:")
        text_label.pack(side="left", padx=2)

        self.text_display_var = tk.StringVar()
        self.text_display_var.set("None")  # Default value

        self.text_display_dropdown = ttk.Combobox(
            left_group,
            textvariable=self.text_display_var,
            values=["None", "Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"],
            width=15,
            state="readonly"
        )
        self.text_display_dropdown.pack(side="left", padx=2)
        self.text_display_dropdown.bind('<<ComboboxSelected>>', self.on_text_display_change)

        # Font size control buttons
        font_size_label = tk.Label(left_group, text="Font:")
        font_size_label.pack(side="left", padx=(10, 2))
        
        self.font_decrease_button = tk.Button(left_group, text="-", command=self.decrease_font_size, width=2)
        self.font_decrease_button.pack(side="left", padx=1)
        
        self.font_increase_button = tk.Button(left_group, text="+", command=self.increase_font_size, width=2)
        self.font_increase_button.pack(side="left", padx=1)

        self.chunking_strategy_var = tk.StringVar()

        # Right group elements (navigation)
        doc_num_label = tk.Label(right_group, text="Document Number:") # Added Document Number label
        doc_num_label.pack(side="left", padx=(5, 2)) # Added Document Number label

        self.button1 = tk.Button(right_group, text="<<", command=lambda: self.navigate_images(-2))
        self.button1.pack(side="left", padx=2)

        self.button2 = tk.Button(right_group, text="<", command=lambda: self.navigate_images(-1))
        self.button2.pack(side="left", padx=2)

        self.page_counter_var = tk.StringVar()
        self.page_counter_var.set("0 / 0")
        page_counter_label = tk.Label(right_group, textvariable=self.page_counter_var)
        page_counter_label.pack(side="left", padx=2)

        self.button4 = tk.Button(right_group, text=">", command=lambda: self.navigate_images(1))
        self.button4.pack(side="left", padx=2)

        self.button5 = tk.Button(right_group, text=">>", command=lambda: self.navigate_images(2))
        self.button5.pack(side="left", padx=2)

        # --- Main and Bottom Frames ---
        self.main_frame = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_frame.grid(row=1, column=0, sticky="nsew")

        self.text_display = self.create_text_widget(self.main_frame, "File to Edit", state="normal")

        self.image_display = tk.Canvas(self.main_frame, borderwidth=2, relief="groove")
        self.image_handler = ImageHandler(self.image_display, self)

        self.main_frame.add(self.text_display)
        self.main_frame.add(self.image_display)

        # --- Bottom Bar Frame ---
        self.bottom_bar_frame = tk.Frame(self)
        self.bottom_bar_frame.grid(row=2, column=0, sticky="nsew")
        self.bottom_bar_frame.grid_columnconfigure(1, weight=1) # Make middle column expandable

        # Create groups in the bottom bar frame
        bottom_left_group = tk.Frame(self.bottom_bar_frame)
        bottom_left_group.grid(row=0, column=0, sticky="w", padx=5)

        # Placeholder to keep the bottom bar visible
        placeholder_label = tk.Label(self.bottom_bar_frame, text=" ")
        placeholder_label.grid(row=0, column=1, sticky="ew")

        bottom_right_group = tk.Frame(self.bottom_bar_frame)
        bottom_right_group.grid(row=0, column=2, sticky="e", padx=(0, 5))

        # Bottom Left group elements (Relevance)
        self.relevance_label = tk.Label(bottom_left_group, text="Relevance:") # Made instance variable
        self.relevance_label.pack(side="left", padx=2)

        self.relevance_dropdown = ttk.Combobox(
            bottom_left_group,
            textvariable=self.relevance_var,
            values=["", "Relevant", "Partially Relevant", "Irrelevant", "Uncertain"], # Added blank default
            width=15,
            state="readonly"
        )
        self.relevance_dropdown.pack(side="left", padx=2)
        self.relevance_dropdown.bind('<<ComboboxSelected>>', self.on_relevance_change)

        # Relevance Navigation Buttons 
        self.relevant_back_button = tk.Button(bottom_left_group, text="Previous", command=lambda: self.navigate_relevant(-1)) # Changed text
        self.relevant_back_button.pack(side="left", padx=(5, 2)) # Add padding to separate from dropdown

        self.relevant_forward_button = tk.Button(bottom_left_group, text="Next", command=lambda: self.navigate_relevant(1)) # Changed text
        self.relevant_forward_button.pack(side="left", padx=2)

        # Bottom Right group elements (Document Page Navigation)
        self.doc_page_label = tk.Label(bottom_right_group, text="Document Page:") # Made instance variable
        self.doc_page_label.pack(side="left", padx=(5, 2))

        self.doc_button1 = tk.Button(bottom_right_group, text="<<", command=lambda: self.document_page_nav(-2))
        self.doc_button1.pack(side="left", padx=2)

        self.doc_button2 = tk.Button(bottom_right_group, text="<", command=lambda: self.document_page_nav(-1))
        self.doc_button2.pack(side="left", padx=2)

        self.doc_page_counter_var = tk.StringVar() # Variable for doc page counter
        self.doc_page_counter_var.set("0 / 0")
        self.doc_page_counter_label = tk.Label(bottom_right_group, textvariable=self.doc_page_counter_var) # Made instance variable
        self.doc_page_counter_label.pack(side="left", padx=2)

        self.doc_button4 = tk.Button(bottom_right_group, text=">", command=lambda: self.document_page_nav(1))
        self.doc_button4.pack(side="left", padx=2)

        self.doc_button5 = tk.Button(bottom_right_group, text=">>", command=lambda: self.document_page_nav(2))
        self.doc_button5.pack(side="left", padx=2)

        self.initialize_temp_directory()
        if not hasattr(self, 'project_directory') or not self.project_directory:
            self.project_directory = self.temp_directory

        # Update Chunking Dropdown with Actual Presets
        preset_names = [p['name'] for p in self.settings.chunk_text_presets] if self.settings.chunk_text_presets else []
        if preset_names:
            self.chunking_strategy_var.set(preset_names[0])
        else:
            self.chunking_strategy_var.set("No Presets")

        # Initialize the Find and Replace window
        self.find_replace = FindReplace(
            parent=self,
            text_display=self.text_display,
            main_df=self.main_df,
            navigate_callback=self.find_replace_navigate,  # Use a dedicated method
            get_page_counter=lambda: self.page_counter,
            get_main_df_callback=lambda: self.main_df,
            text_display_var=self.text_display_var  # Pass the StringVar
        )

        # Initialize the API handler
        self.api_handler = APIHandler(
            self.settings.openai_api_key,
            self.settings.anthropic_api_key,
            self.settings.google_api_key,
            self
        )

        # Initialize the Progress Bar
        self.progress_bar = ProgressBar(self)

        # Initialize ProjectIO
        self.project_io = ProjectIO(self)

        # Initialize the export manager
        self.export_manager = ExportManager(self)

        # Initialize the AI Functions Handler <<<<<<<<<<<<<<<<<<<< NEW
        self.ai_functions_handler = AIFunctionsHandler(self)

        # Initialize the Names and Places Handler <<<<<<<<<<<<<<<< NEW
        self.names_places_handler = NamesAndPlacesHandler(self)

        # Initialize the Highlight Handler <<<<<<<<<<<<<<<<<<<<<< NEW
        self.highlight_handler = HighlightHandler(self)

        # Variables to store last selected dropdown values <--- ADDED
        self.last_selected_format_preset = None
        self.last_selected_chunking_strategy = None
        self.last_selected_htr_preset = None
        # <--- END ADDED

        # Configure highlight tags
        self.text_display.tag_config("name_highlight", background="lightblue")
        self.text_display.tag_config("place_highlight", background="wheat1")
        self.text_display.tag_config("change_highlight", background="lightgreen")
        self.text_display.tag_config("error_highlight", background="cyan")

        # Create menus and key bindings
        self.create_menus()
        self.create_key_bindings()
        self.bind_key_universal_commands(self.text_display)

        # Enable drag and drop
        self.enable_drag_and_drop()

        # Apply initial visibility settings
        self.toggle_relevance_visibility() # Call this to set initial state based on show_relevance
        self.toggle_page_nav_visibility()
        # self.toggle_relevance_nav_visibility() # REMOVED

        # Add initialization of data_operations after other handlers are initialized
        self.data_operations = DataOperations(self)

# GUI Setup

    def create_menus(self):
        # Create menu bar
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # File menu

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New Project", command=self.project_io.create_new_project)
        self.file_menu.add_command(label="Open Project", command=self.project_io.open_project)
        
        # Create Recent Projects submenu
        self.recent_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Open Recent Projects", menu=self.recent_menu)
        self.update_recent_projects_menu()
        
        self.file_menu.add_command(label="Save Project", command=self.project_io.save_project)
        self.file_menu.add_command(label="Save Project As", command=self.project_io.save_project_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Import PDF...", command=self.import_pdf)
        self.file_menu.add_command(label="Import Images from Folder...", command=lambda: self.open_folder("Images without Text"))
        self.file_menu.add_command(label="Import Text and Images...", command=lambda: self.open_folder("With Text"))
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Export Text...", command=self.export_manager.export_menu)
        self.file_menu.add_command(label="Export CSV...", command=self.export_manager.show_csv_export_options)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Settings", command=self.create_settings_window)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_closing)

        # Edit Menu

        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(label="Find and Replace", command=self.find_and_replace)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Undo", command=self.undo)
        self.edit_menu.add_command(label="Redo", command=self.redo)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Cut",
            command=lambda: self.text_display.event_generate("<<Cut>>"))
        self.edit_menu.add_command(label="Copy",
            command=lambda: self.text_display.event_generate("<<Copy>>"))
        self.edit_menu.add_command(label="Paste",
        command=lambda: self.text_display.event_generate("<<Paste>>"))
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Rotate Image Clockwise", command=lambda: self.rotate_image(-90))
        self.edit_menu.add_command(label="Rotate Image Counter-clockwise", command=lambda: self.rotate_image(90))
        self.edit_menu.add_command(label="Auto-get Rotation (Current Page)", command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag="Current Page", ai_job="Auto_Rotate"))
        self.edit_menu.add_command(label="Auto-get Rotation (All Pages)", command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Auto_Rotate"))
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Revert Current Page", command=self.data_operations.revert_current_page)
        self.edit_menu.add_command(label="Revert All Pages", command=self.data_operations.revert_all_pages)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Delete Current Image", command=self.data_operations.delete_current_image)

        # Process Menu

        self.process_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Process", menu=self.process_menu)
        self.process_mode = tk.StringVar(value="All Pages")  # Default to All Pages

        # Mode selection submenu
        mode_menu = tk.Menu(self.process_menu, tearoff=0)

        # Page mode options
        mode_menu.add_radiobutton(label="Current Page", variable=self.process_mode, value="Current Page")
        mode_menu.add_radiobutton(label="All Pages", variable=self.process_mode, value="All Pages")
        mode_menu.add_separator()

        # Skip/Redo toggle as checkbutton
        mode_menu.add_checkbutton(
            label="Skip Completed Pages",
            variable=self.skip_completed_pages,
            onvalue=True,
            offvalue=False
        )

        self.process_menu.add_cascade(label="Processing Mode", menu=mode_menu)
        self.process_menu.add_separator()
        self.process_menu.add_command(label="Recognize Text",
                                     command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="HTR"))
        self.process_menu.add_command(label="Correct Text",
                                     command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="Correct_Text"))
        self.process_menu.add_command(label="Format Text",
                                     command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="Format_Text"))
        self.process_menu.add_separator()
        self.process_menu.add_command(label="Translate Text",
                                     command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="Translation"))
        self.process_menu.add_separator()
        self.process_menu.add_command(label="Find Errors",
                                     command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="Identify_Errors"))

        # Document Menu

        self.document_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Highlights", menu=self.document_menu)
        self.document_menu.add_separator()
        self.highlight_names_var = tk.BooleanVar()
        self.highlight_places_var = tk.BooleanVar()
        self.highlight_changes_var = tk.BooleanVar()
        self.highlight_errors_var = tk.BooleanVar()
        self.document_menu.add_checkbutton(label="Highlight Names",
                                        variable=self.highlight_names_var,
                                        command=self.toggle_highlight_options)
        self.document_menu.add_checkbutton(label="Highlight Places",
                                        variable=self.highlight_places_var,
                                        command=self.toggle_highlight_options)
        self.document_menu.add_checkbutton(label="Highlight Changes",
                                        variable=self.highlight_changes_var,
                                        command=self.toggle_highlight_options)
        self.document_menu.add_checkbutton(label="Highlight Errors",
                                        variable=self.highlight_errors_var,
                                        command=self.toggle_highlight_options)

        # Tools Menu

        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(label="Edit Current Image", command=self.edit_single_image)
        self.tools_menu.add_command(label="Edit All Images", command=self.edit_all_images)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(
            label="Get Names and Places",
            command=lambda: self.ai_functions_handler.ai_function(all_or_one_flag=self.process_mode.get(), ai_job="Get_Names_and_Places")
        )
        self.tools_menu.add_separator()
        self.tools_menu.add_command(
            label="Edit Names & Places",
            command=self.names_places_handler.initiate_collation_and_show_window
        )
        self.tools_menu.add_separator()
        self.tools_menu.add_command(
            label="Identify Document Separators",
            command=lambda: self.create_chunk_text_window(self.process_mode.get())
        )
        self.tools_menu.add_command(
            label="Apply Document Separation",
            command=self.apply_document_separation
        )
        self.tools_menu.add_separator()
        self.tools_menu.add_command(
            label="Find Relevant Documents",
            command=self.create_find_relevant_documents_window # <-- uses ai_functions_handler internally
        )

    def create_key_bindings(self):
        # Navigation bindings
        self.bind("<Control-Home>", lambda event: self.navigate_images(-2))
        self.bind("<Control-Left>", lambda event: self.navigate_images(-1))
        self.bind("<Control-Right>", lambda event: self.navigate_images(1))
        self.bind("<Control-End>", lambda event: self.navigate_images(2))

        # Rotation bindings
        self.bind("<Control-bracketright>", lambda event: self.rotate_image(90))
        self.bind("<Control-bracketleft>", lambda event: self.rotate_image(-90))

        # Project management bindings
        self.bind("<Control-n>", lambda event: self.project_io.create_new_project())
        self.bind("<Control-o>", lambda event: self.project_io.open_project())
        self.bind("<Control-s>", lambda event: self.project_io.save_project())

        # Edit bindings
        self.bind("<Control-z>", self.undo)
        self.bind("<Control-y>", self.redo)

        # Find and Replace bindings
        self.bind("<Control-f>", lambda event: self.find_and_replace())

        # Revert bindings
        self.bind("<Control-r>", lambda event: self.data_operations.revert_current_page()
)
        self.bind("<Control-Shift-r>", lambda event: self.data_operations.revert_all_pages())

        # Text display toggle binding
        self.bind("<Control-Tab>", lambda event: self.toggle_text())

        # Image management bindings
        self.bind("<Control-d>", lambda event: self.data_operations.delete_current_image())
        self.bind("<Control-i>", lambda event: self.edit_single_image())
        self.bind("<Control-Shift-i>", lambda event: self.edit_all_images())

        # Visibility toggle bindings
        self.bind("<Alt-n>", lambda event: self.toggle_nav_display())

        # AI function bindings - Updated calls to use ai_functions_handler
        self.bind("<Control-1>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="Current Page", ai_job="HTR"))
        self.bind("<Control-Shift-1>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="HTR"))
        self.bind("<Control-2>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="Current Page", ai_job="Correct_Text"))
        self.bind("<Control-Shift-2>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Correct_Text"))
        self.bind("<Control-t>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="Current Page", ai_job="Translation"))
        self.bind("<Control-Shift-t>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Translation"))

        # Add key binding for Format_Text
        self.bind("<Control-5>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="Current Page", ai_job="Format_Text"))
        self.bind("<Control-Shift-5>", lambda event: self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Format_Text"))

        # Add key bindings for Chunk_Text
        self.bind("<Control-3>", lambda event: self.create_chunk_text_window("Current Page"))
        self.bind("<Control-Shift-3>", lambda event: self.create_chunk_text_window("All Pages"))

        # Add key binding for document separation
        self.bind("<Control-4>", lambda event: self.create_document_separation_options_window())

        # Mouse bindings
        self.image_display.bind("<Control-MouseWheel>", self.image_handler.zoom)
        self.image_display.bind("<MouseWheel>", self.image_handler.scroll)
        self.image_display.bind("<ButtonPress-1>", self.image_handler.start_pan)
        self.image_display.bind("<B1-Motion>", self.image_handler.pan)

        # Document Page Navigation Bindings (Example - adjust as needed)
        self.bind("<Alt-Left>", lambda event: self.document_page_nav(-1))
        self.bind("<Alt-Right>", lambda event: self.document_page_nav(1))

        # Font size adjustment bindings
        self.bind("<Control-plus>", lambda event: self.increase_font_size())
        self.bind("<Control-equal>", lambda event: self.increase_font_size())  # For keyboards where + requires shift
        self.bind("<Control-minus>", lambda event: self.decrease_font_size())

    def create_image_widget(self, frame, image_path, state):
        # Load the image
        original_image = Image.open(image_path)
        self.photo_image = ImageTk.PhotoImage(original_image)

        # Create a canvas and add the image to it
        self.canvas = tk.Canvas(frame, borderwidth=2, relief="groove")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.grid(sticky="nsew")

        # Bind zoom and scroll events to the image_handler methods
        self.canvas.bind("<Control-MouseWheel>", self.image_handler.zoom)
        self.canvas.bind("<MouseWheel>", self.image_handler.scroll)

        return self.canvas

    def create_text_widget(self, frame, label_text, state):
        # Create a Text widget to display the contents of the selected file
        text_display = tk.Text(frame, wrap="word", state=state, undo=True)
        # Use the persistent font size
        text_display.config(font=("Calibri", self.text_font_size))

        text_display.grid(sticky="nsew")

        return text_display

    def bind_key_universal_commands(self, text_widget):
        text_widget.bind('<Control-h>', self.find_and_replace)
        text_widget.bind('<Control-f>', self.find_and_replace)
        text_widget.bind('<Control-z>', self.undo)
        text_widget.bind('<Control-y>', self.redo)

# Create Secondary Windows

    def create_find_relevant_documents_window(self):
        """Create a window for finding relevant documents using AI analysis"""
        # Create the window
        relevance_window = tk.Toplevel(self)
        relevance_window.title("Find Relevant Documents")
        relevance_window.geometry("600x400")
        relevance_window.grab_set()  # Make window modal

        # Message explaining purpose
        message_label = tk.Label(relevance_window,
            text="Describe what makes a document relevant to your research:",
            font=("Calibri", 12))
        message_label.pack(pady=15)

        # Create a large text box for the relevance criteria
        criteria_frame = tk.Frame(relevance_window)
        criteria_frame.pack(fill="both", expand=True, padx=10, pady=5)

        criteria_text = tk.Text(criteria_frame, wrap="word", height=10, undo=True)
        criteria_text.pack(fill="both", expand=True, side="left")

        # Add scrollbar
        scrollbar = tk.Scrollbar(criteria_frame, command=criteria_text.yview)
        scrollbar.pack(side="right", fill="y")
        criteria_text.config(yscrollcommand=scrollbar.set)

        # Create the text source selection dropdown
        dropdown_frame = tk.Frame(relevance_window)
        dropdown_frame.pack(pady=10)

        source_label = tk.Label(dropdown_frame, text="Text Source:")
        source_label.pack(side="left", padx=5)

        # Create a StringVar for the text source dropdown
        self.relevance_source_var = tk.StringVar()

        # Create text source options based on available data
        source_options = []
        if not self.main_df.empty:
            row_index = self.page_counter if self.page_counter < len(self.main_df) else 0 # Use 0 if out of bounds
            if row_index < len(self.main_df):
                row = self.main_df.iloc[row_index]
                if pd.notna(row.get('Original_Text')) and row.get('Original_Text', "").strip():
                    source_options.append("Original_Text")
                if pd.notna(row.get('Corrected_Text')) and row.get('Corrected_Text', "").strip():
                    source_options.append("Corrected_Text")
                if pd.notna(row.get('Formatted_Text')) and row.get('Formatted_Text', "").strip():
                    source_options.append("Formatted_Text")
                if pd.notna(row.get('Translation')) and row.get('Translation', "").strip():
                    source_options.append("Translation")
                if pd.notna(row.get('Separated_Text')) and row.get('Separated_Text', "").strip():
                    source_options.append("Separated_Text")

        # If no options, add some defaults
        if not source_options:
            source_options = ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]

        # Create the text source dropdown
        source_dropdown = ttk.Combobox(dropdown_frame,
                                    textvariable=self.relevance_source_var,
                                    values=source_options,
                                    state="readonly",
                                    width=20)
        source_dropdown.pack(side="left", padx=5)

        # Set a default value - prefer formatted or corrected text
        if "Formatted_Text" in source_options:
            self.relevance_source_var.set("Formatted_Text")
        elif "Corrected_Text" in source_options:
            self.relevance_source_var.set("Corrected_Text")
        elif source_options:
            self.relevance_source_var.set(source_options[0])

        # Mode selection for All Pages or Current Page
        mode_frame = tk.Frame(relevance_window)
        mode_frame.pack(pady=10)

        self.relevance_mode_var = tk.StringVar(value="All Pages")

        mode_label = tk.Label(mode_frame, text="Process:")
        mode_label.pack(side="left", padx=5)

        current_radio = tk.Radiobutton(mode_frame, text="Current Page", variable=self.relevance_mode_var, value="Current Page")
        current_radio.pack(side="left", padx=5)

        all_radio = tk.Radiobutton(mode_frame, text="All Pages", variable=self.relevance_mode_var, value="All Pages")
        all_radio.pack(side="left", padx=5)

        # Buttons at the bottom
        button_frame = tk.Frame(relevance_window)
        button_frame.pack(pady=20)

        # Function to handle OK button
        def on_ok():
            criteria_text_content = criteria_text.get("1.0", "end-1c").strip()
            if not criteria_text_content:
                messagebox.showwarning("Warning", "Please enter relevance criteria.")
                return

            selected_source = self.relevance_source_var.get()
            mode = self.relevance_mode_var.get()

            # Close the window
            relevance_window.destroy()

            # Process the relevance search using the handler
            self.ai_functions_handler.process_relevance_search(criteria_text_content, selected_source, mode)

        # Function to handle Cancel button
        def on_cancel():
            relevance_window.destroy()

        # Create buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
        ok_button.pack(side="left", padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
        cancel_button.pack(side="left", padx=10)

        # Center the window
        relevance_window.update_idletasks()
        width = relevance_window.winfo_width()
        height = relevance_window.winfo_height()
        x = (relevance_window.winfo_screenwidth() // 2) - (width // 2)
        y = (relevance_window.winfo_screenheight() // 2) - (height // 2)
        relevance_window.geometry(f'{width}x{height}+{x}+{y}')

        # Bind keyboard shortcuts
        relevance_window.bind("<Return>", lambda event: on_ok())
        relevance_window.bind("<Escape>", lambda event: on_cancel())

        # Give focus to the text widget
        criteria_text.focus_set()

        # Wait for the window to be closed
        self.wait_window(relevance_window)

    def create_text_source_window(self, all_or_one_flag, ai_job):
        """
        Creates a window for selecting text source before running AI functions
        """
        # Create the window
        source_window = tk.Toplevel(self)
        source_window.title(f"Select {'Format Preset and ' if ai_job == 'Format_Text' else ''}Text Source for {ai_job.replace('_', ' ')}")
        source_window.geometry("400x300" if ai_job == "Format_Text" else "400x250")
        source_window.grab_set()  # Make window modal

        # Message explaining purpose
        message_label = tk.Label(source_window,
            text=f"Select {'the format preset and ' if ai_job == 'Format_Text' else ''}the text source to process:",
            font=("Calibri", 12))
        message_label.pack(pady=15)

        # Add format preset selection for Format_Text
        format_frame = None
        additional_info_text = None  # Will hold the text widget for additional info
        if ai_job == "Format_Text":
            format_frame = tk.Frame(source_window)
            format_frame.pack(pady=10)

            format_label = tk.Label(format_frame, text="Format Preset:")
            format_label.pack(side="left", padx=5)

            # Create a StringVar for the format preset dropdown
            self.format_preset_var = tk.StringVar()

            # Get available format presets
            format_options = [preset.get('name', f"Preset {i+1}") for i, preset in enumerate(self.settings.format_presets)]

            # Create the format preset dropdown
            format_dropdown = ttk.Combobox(format_frame,
                                         textvariable=self.format_preset_var,
                                         values=format_options,
                                         state="readonly",
                                         width=20)
            format_dropdown.pack(side="left", padx=5)

            # Set default to last selected format preset if available, else first <--- MODIFIED
            if self.last_selected_format_preset and self.last_selected_format_preset in format_options:
                self.format_preset_var.set(self.last_selected_format_preset)
            elif format_options: # Fallback to first if no last selection or invalid
                self.format_preset_var.set(format_options[0])
            # <--- END MODIFIED

            # --- Add additional info text box at the bottom ---
            additional_info_frame = tk.Frame(source_window)
            additional_info_frame.pack(fill="x", padx=10, pady=(10, 0))
            additional_info_label = tk.Label(additional_info_frame, text="Add any additional information for the model about your document to use for formatting:")
            additional_info_label.pack(anchor="w")
            self.format_additional_info_text = tk.Text(additional_info_frame, height=4, width=40, wrap="word")
            self.format_additional_info_text.pack(fill="x", expand=True)
        else:
            self.format_additional_info_text = None

        # Create the text source selection dropdown
        dropdown_frame = tk.Frame(source_window)
        dropdown_frame.pack(pady=10)

        source_label = tk.Label(dropdown_frame, text="Text Source:")
        source_label.pack(side="left", padx=5)

        # Create a StringVar for the text source dropdown
        self.text_source_var = tk.StringVar()

        # Create text source options based on available data
        source_options = []
        if not self.main_df.empty and self.page_counter < len(self.main_df):
            row_index = self.page_counter
            row = self.main_df.iloc[row_index]
            if pd.notna(row.get('Original_Text')) and row.get('Original_Text', "").strip():
                source_options.append("Original_Text")
            if pd.notna(row.get('Corrected_Text')) and row.get('Corrected_Text', "").strip():
                source_options.append("Corrected_Text")
            if pd.notna(row.get('Formatted_Text')) and row.get('Formatted_Text', "").strip():
                source_options.append("Formatted_Text")
            if pd.notna(row.get('Translation')) and row.get('Translation', "").strip():
                source_options.append("Translation")
            if pd.notna(row.get('Separated_Text')) and row.get('Separated_Text', "").strip():
                source_options.append("Separated_Text")


        # If no options, add some defaults
        if not source_options:
            source_options = ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]

        # Create the text source dropdown
        source_dropdown = ttk.Combobox(dropdown_frame,
                                    textvariable=self.text_source_var,
                                    values=source_options,
                                    state="readonly",
                                    width=20)
        source_dropdown.pack(side="left", padx=5)

        # Define the desired hierarchy for default selection
        hierarchy = ["Separated_Text", "Translation", "Formatted_Text", "Corrected_Text", "Original_Text"]

        # Set the default based on the hierarchy
        default_set = False
        for text_type in hierarchy:
            if text_type in source_options:
                self.text_source_var.set(text_type)
                default_set = True
                break
        # Fallback if no hierarchical match (should not happen if options exist)
        if not default_set and source_options:
            self.text_source_var.set(source_options[0])

        # Buttons frame
        button_frame = tk.Frame(source_window)
        button_frame.pack(pady=20)

        # Function to handle OK button
        def on_ok():
            # For Format_Text, store the additional info and update last selection <--- MODIFIED
            if ai_job == "Format_Text":
                if self.format_additional_info_text is not None:
                    self.format_additional_info = self.format_additional_info_text.get("1.0", "end-1c").strip()
                # Store the selected format preset <--- ADDED
                if hasattr(self, 'format_preset_var'): # Ensure var exists
                    self.last_selected_format_preset = self.format_preset_var.get()
                # <--- END ADDED
            else:
                self.format_additional_info = None
            # <--- END MODIFIED
            # Close the window
            source_window.destroy()
            # Run the AI function with the selected parameters using the handler
            self.ai_functions_handler.process_ai_with_selected_source(all_or_one_flag, ai_job)

        # Function to handle Cancel button
        def on_cancel():
            source_window.destroy()

        # Create buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
        ok_button.pack(side="left", padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
        cancel_button.pack(side="left", padx=10)

        # Center the window on the screen
        source_window.update_idletasks()
        width = source_window.winfo_width()
        height = source_window.winfo_height()
        x = (source_window.winfo_screenwidth() // 2) - (width // 2)
        y = (source_window.winfo_screenheight() // 2) - (height // 2)
        source_window.geometry(f'{width}x{height}+{x}+{y}')

        # Wait for the window to be closed
        self.wait_window(source_window)

    def create_htr_preset_window(self, all_or_one_flag, ai_job):
        """
        Creates a window for selecting HTR preset before running HTR
        """
        # Create the window
        htr_window = tk.Toplevel(self)
        htr_window.title("Select HTR Preset")
        htr_window.geometry("400x200")
        htr_window.grab_set()  # Make window modal

        # Message explaining purpose
        message_label = tk.Label(htr_window,
            text="Select the HTR preset to use for text recognition:",
            font=("Calibri", 12))
        message_label.pack(pady=15)

        # HTR preset selection dropdown
        preset_frame = tk.Frame(htr_window)
        preset_frame.pack(pady=10)

        preset_label = tk.Label(preset_frame, text="HTR Preset:")
        preset_label.pack(side="left", padx=5)

        # Create a StringVar for the HTR preset dropdown
        self.htr_preset_var = tk.StringVar()

        # Get available HTR presets from transcription_presets
        htr_options = [preset.get('name', f"Preset {i+1}") for i, preset in enumerate(self.settings.transcription_presets)]

        # Create the HTR preset dropdown
        htr_dropdown = ttk.Combobox(preset_frame,
                                   textvariable=self.htr_preset_var,
                                   values=htr_options,
                                   state="readonly",
                                   width=30)
        htr_dropdown.pack(side="left", padx=5)

        # Set default to last selected HTR preset if available, else look for "HTR" preset, else first
        if self.last_selected_htr_preset and self.last_selected_htr_preset in htr_options:
            self.htr_preset_var.set(self.last_selected_htr_preset)
        elif "HTR" in htr_options:
            self.htr_preset_var.set("HTR")
        elif htr_options:
            self.htr_preset_var.set(htr_options[0])

        # Buttons frame
        button_frame = tk.Frame(htr_window)
        button_frame.pack(pady=20)

        # Function to handle OK button
        def on_ok():
            # Store the selected HTR preset
            if hasattr(self, 'htr_preset_var'):
                self.last_selected_htr_preset = self.htr_preset_var.get()
            # Close the window
            htr_window.destroy()
            # Run the AI function with the selected parameters using the handler
            self.ai_functions_handler.process_htr_with_selected_preset(all_or_one_flag, ai_job)

        # Function to handle Cancel button
        def on_cancel():
            htr_window.destroy()

        # Create buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
        ok_button.pack(side="left", padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
        cancel_button.pack(side="left", padx=10)

        # Center the window on the screen
        htr_window.update_idletasks()
        width = htr_window.winfo_width()
        height = htr_window.winfo_height()
        x = (htr_window.winfo_screenwidth() // 2) - (width // 2)
        y = (htr_window.winfo_screenheight() // 2) - (height // 2)
        htr_window.geometry(f'{width}x{height}+{x}+{y}')

        # Bind keyboard shortcuts
        htr_window.bind("<Return>", lambda event: on_ok())
        htr_window.bind("<Escape>", lambda event: on_cancel())

        # Give focus to the dropdown
        htr_dropdown.focus_set()

        # Wait for the window to be closed
        self.wait_window(htr_window)

    def create_chunk_text_window(self, all_or_one_flag):
        """
        Creates a window for selecting document type before running Chunk_Text
        """
        # Create the window
        chunk_window = tk.Toplevel(self)
        chunk_window.title("Select Document Type")
        chunk_window.geometry("400x300")  # Made taller to accommodate new dropdown
        chunk_window.grab_set()  # Make window modal

        # Message explaining purpose
        message_label = tk.Label(chunk_window,
            text="Select the document type for text chunking:",
            font=("Calibri", 12))
        message_label.pack(pady=15)

        # Create the chunking strategy dropdown in this window
        dropdown_frame = tk.Frame(chunk_window)
        dropdown_frame.pack(pady=10)

        chunking_label = tk.Label(dropdown_frame, text="Document Type:")
        chunking_label.pack(side="left", padx=5)

        # Create a new StringVar for this window's dropdown
        window_chunking_var = tk.StringVar()

        # If there's already a selected strategy, use it as default
        if hasattr(self, 'chunking_strategy_var') and self.chunking_strategy_var.get():
            window_chunking_var.set(self.chunking_strategy_var.get())

        # Get preset names for the dropdown
        preset_names = [p['name'] for p in self.settings.chunk_text_presets] if self.settings.chunk_text_presets else []

        # Create the dropdown
        chunking_dropdown = ttk.Combobox(dropdown_frame,
                                      textvariable=window_chunking_var,
                                      values=preset_names,
                                      state="readonly",
                                      width=30)
        chunking_dropdown.pack(side="left", padx=5)

        # Set a default value if available, prioritizing last selection <--- MODIFIED
        if self.last_selected_chunking_strategy and self.last_selected_chunking_strategy in preset_names:
            window_chunking_var.set(self.last_selected_chunking_strategy)
        elif preset_names and not window_chunking_var.get(): # Fallback to first if no last selection or current isn't set
            window_chunking_var.set(preset_names[0])
        # <--- END MODIFIED

        # Add a new frame for text source selection
        text_source_frame = tk.Frame(chunk_window)
        text_source_frame.pack(pady=15)

        text_source_label = tk.Label(text_source_frame, text="Text Source:")
        text_source_label.pack(side="left", padx=5)

        # Create a StringVar for the text source dropdown
        self.chunk_text_source_var = tk.StringVar()

        # Create text source options based on available data
        source_options = []
        if not self.main_df.empty and self.page_counter < len(self.main_df):
            row_index = self.page_counter
            row = self.main_df.iloc[row_index]
            if pd.notna(row.get('Original_Text')) and row.get('Original_Text', "").strip():
                source_options.append("Original_Text")
            if pd.notna(row.get('Corrected_Text')) and row.get('Corrected_Text', "").strip():
                source_options.append("Corrected_Text")
            if pd.notna(row.get('Formatted_Text')) and row.get('Formatted_Text', "").strip():
                source_options.append("Formatted_Text")
            if pd.notna(row.get('Translation')) and row.get('Translation', "").strip():
                source_options.append("Translation")
            if pd.notna(row.get('Separated_Text')) and row.get('Separated_Text', "").strip():
                source_options.append("Separated_Text")

        # If no options, add some defaults
        if not source_options:
            source_options = ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]

        # Create the text source dropdown (populate with available options)
        text_source_dropdown = ttk.Combobox(text_source_frame,
                                         textvariable=self.chunk_text_source_var,
                                         values=source_options,
                                         state="readonly",
                                         width=20)
        text_source_dropdown.pack(side="left", padx=5)

        # Define the desired hierarchy for default selection
        hierarchy = ["Separated_Text", "Translation", "Formatted_Text", "Corrected_Text", "Original_Text"]

        # Set the default based on the hierarchy
        default_set = False
        for text_type in hierarchy:
            if text_type in source_options:
                self.chunk_text_source_var.set(text_type)
                default_set = True
                break
        # Fallback if no hierarchical match (should not happen if options exist)
        if not default_set and source_options:
            self.chunk_text_source_var.set(source_options[0])

        # Buttons frame
        button_frame = tk.Frame(chunk_window)
        button_frame.pack(pady=20)

        # Function to handle OK button
        def on_ok():
            # Set the main window's chunking strategy variable
            self.chunking_strategy_var.set(window_chunking_var.get())
            # Store the selected chunking strategy <--- ADDED
            self.last_selected_chunking_strategy = window_chunking_var.get()
            # <--- END ADDED
            # Close the window
            chunk_window.destroy()
            # Run the AI function with the selected parameters using the handler
            self.ai_functions_handler.ai_function(all_or_one_flag=all_or_one_flag, ai_job="Chunk_Text")

        # Function to handle Cancel button
        def on_cancel():
            chunk_window.destroy()

        # Create buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
        ok_button.pack(side="left", padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
        cancel_button.pack(side="left", padx=10)

        # Center the window on the screen
        chunk_window.update_idletasks()
        width = chunk_window.winfo_width()
        height = chunk_window.winfo_height()
        x = (chunk_window.winfo_screenwidth() // 2) - (width // 2)
        y = (chunk_window.winfo_screenheight() // 2) - (height // 2)
        chunk_window.geometry(f'{width}x{height}+{x}+{y}')

        # Wait for the window to be closed
        self.wait_window(chunk_window)

    def create_document_separation_options_window(self):
        """Opens a window with document separation options."""
        try:
            # First check if any separators exist in the document
            has_separators = False
            for index, row in self.main_df.iterrows():
                # Check all text columns
                for col in ['Original_Text', 'Corrected_Text', 'Formatted_Text', 'Translation', 'Separated_Text']:
                    if col in self.main_df.columns and pd.notna(row.get(col)) and isinstance(row.get(col), str):
                        # Use regex to look for 5 or more consecutive asterisks (allowing for whitespace)
                        if re.search(r'\*{5,}', row.get(col)):
                            has_separators = True
                            break
                if has_separators:
                    break

            # If no separators found, warn the user and don't open the window
            if not has_separators:
                messagebox.showwarning(
                    "No Separators Found",
                    "No document separators ('*****') were found in any text column. "
                    "Please add separators before applying document separation."
                )
                return

            # Call the simplified document separation function
            self.apply_document_separation()

        finally:
            # Make sure buttons are properly enabled
            if self.button1['state'] == "disabled" and self.button2['state'] == "disabled":
                self.toggle_button_state()

# Initialize Settings Functions

    def reset_application(self):
        # Clear the main DataFrame using DataOperations
        self.data_operations.initialize_main_df()

        # Reset Flags, Toggles, and Variables
        self.save_toggle = False
        self.find_replace_toggle = False
        self.original_image = None
        self.photo_image = None
        self.pagination_added = False
        self.current_image_path_list = None # <--- ADDED: Explicitly reset image list
        self.current_doc_page_index = 0    # <--- ADDED: Explicitly reset doc page index

        # Reset highlight toggles to False instead of recreating them
        self.highlight_names_var.set(False)
        self.highlight_places_var.set(False)
        self.highlight_changes_var.set(False)
        self.highlight_errors_var.set(False)

        # Reset visibility toggles to default (False for most, True for relevance nav)
        self.show_relevance.set(False)
        self.show_page_nav.set(False)
        self.toggle_relevance_visibility()
        self.toggle_page_nav_visibility()
        # self.toggle_relevance_nav_visibility() # Apply relevance nav reset - REMOVED as visibility tied to relevance section

        # Reset page counter
        self.page_counter = 0

        # Reset flags
        self.save_toggle = False
        self.find_replace_toggle = False

        # Clear text displays
        self.text_display.delete("1.0", tk.END)

        # Clear image display
        self.image_display.delete("all")
        self.current_image_path = None
        self.original_image = None
        self.photo_image = None

        # Reset zoom and pan in image handler
        if hasattr(self, 'image_handler'):
            self.image_handler.current_scale = 1

        # Reset counter
        self.counter_update()

        # Ensure Apply Separation menu item is enabled
        self.update_separation_menu_state("normal")

        # Clear project and image directories
        self.initialize_temp_directory()
        # Reset the page counter
        self.page_counter = 0

        # Clear the find and replace matches DataFrame
        if hasattr(self, 'find_replace'): # Ensure find_replace exists
            self.find_replace.find_replace_matches_df = pd.DataFrame(columns=["Index", "Page"])

        # Reset the dropdown to "None"
        self.text_display_var.set("None")

        # Refresh the display to reflect the cleared state
        self.refresh_display() # <--- ADDED: Call refresh_display

    def initialize_settings(self):
        # Get the appropriate app data directory
        if os.name == 'nt':  # Windows
            app_data = os.path.join(os.environ['APPDATA'], 'TranscriptionPearl')
        else:  # Linux/Mac
            app_data = os.path.join(os.path.expanduser('~'), '.transcriptionpearl')

        # Create the directory if it doesn't exist
        os.makedirs(app_data, exist_ok=True)

        # Define settings file path
        self.settings_file_path = os.path.join(app_data, 'settings.json')

        # Initialize the DataFrame using DataOperations
        self.data_operations.initialize_main_df()

        # Initilize the dataframe to store metadata
        self.compiled_df = pd.DataFrame(columns=["Index", "Document_No", "Document_Type", "Text", "Translation", "Document_Page", "Citation", "Image_Path", "Author", "Correspondent", "Creation_Place", "Correspondent_Place:", "Date", "Places", "People", "Summary", "Temp_Data_Analysis", "Data_Analysis", "Query_Data", "Query_Memory", "Relevance", "Notes"])

        # First set default values
        self.settings.restore_defaults()
        # Update API handler if keys changed
        self.update_api_handler()

        # Check if settings file exists and load it
        if os.path.exists(self.settings_file_path):
            self.settings.load_settings() # Call load_settings on the Settings instance

    def initialize_temp_directory(self):
        """Initialize and manage temporary directories"""
        # Define temp directory paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_directory = os.path.join(base_dir, "util", "temp")
        self.images_directory = os.path.join(self.temp_directory, "images")
        self.edit_temp_directory = os.path.join(self.temp_directory, "edit_temp")

        # Clear existing temp directories
        for dir_path in [self.temp_directory, self.images_directory, self.edit_temp_directory]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to clear temp directory: {e}")
                    self.error_logging(f"Failed to clear temp directory: {e}")

        # Create fresh temp directories
        try:
            for dir_path in [self.temp_directory, self.images_directory, self.edit_temp_directory]:
                os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create temp directories: {e}")
            self.error_logging(f"Failed to create temp directories: {e}")

        # Reset the main DataFrame using DataOperations
        self.data_operations.initialize_main_df()
        self.page_counter = 0

    def create_settings_window(self):
        self.toggle_button_state()
        settings_window = SettingsWindow(self, mode="T_PEARL")
        
    def create_settings_window_to_tab(self, tab_name):
        """Create settings window and open to a specific tab"""
        self.toggle_button_state()
        settings_window = SettingsWindow(self, mode="T_PEARL")
        
        # Force window to take focus
        settings_window.settings_window.attributes('-topmost', True)
        settings_window.settings_window.focus_force()
        
        # Show the specified tab
        settings_window.show_settings(tab_name)
        return settings_window

    def enable_drag_and_drop(self):
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop)

# Navigation Functions

    def navigate_images(self, direction):
        if self.main_df.empty:
            return

        # --- Save current text before navigating ---
        # (Your existing save logic here - seems okay)
        if self.page_counter < len(self.main_df):
            current_display_val = self.main_df.loc[self.page_counter, 'Text_Toggle']
            if current_display_val != "None":
                text = self.data_operations.clean_text(self.text_display.get("1.0", tk.END))
                if current_display_val in ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]:
                    self.main_df.loc[self.page_counter, current_display_val] = text
        # --- End Save ---

        # Store the current display type
        selected_display = self.text_display_var.get()

        # Handle navigation logic (seems okay)
        if abs(direction) == 2:
            if direction < 0: self.page_counter = 0
            else: self.page_counter = len(self.main_df) - 1
        else:
            new_counter = self.page_counter + direction
            if new_counter < 0: new_counter = 0
            elif new_counter >= len(self.main_df): new_counter = len(self.main_df) - 1
            self.page_counter = new_counter

        # *** IMPORTANT: Reset document page index when moving between main documents ***
        self.current_doc_page_index = 0
        # ****************************************************************************

        # Call refresh_display which now handles path logic and loading
        self.refresh_display()

        # Update the Text_Toggle in the DataFrame AFTER potentially changing pages
        # Ensure page_counter is valid before accessing main_df
        if not self.main_df.empty and self.page_counter < len(self.main_df) and selected_display != "None":
            self.main_df.at[self.page_counter, 'Text_Toggle'] = selected_display
        # Load text is now called within refresh_display
        # self.load_text() # Removed - handled by refresh_display
        # self.counter_update() # Removed - handled by load_text inside refresh_display

    def counter_update(self):
        total_images = len(self.main_df) -1 # Index is 0-based

        if total_images >= 0:
            self.page_counter_var.set(f"{self.page_counter + 1} / {total_images + 1}")
        else:
            self.page_counter_var.set("0 / 0")

    def find_replace_navigate(self, direction):
        """Special navigation method for find/replace functionality"""
        if hasattr(self, 'find_replace') and hasattr(self.find_replace, 'link_nav'):

            # Set the page counter
            self.page_counter = self.find_replace.link_nav

            try:
                # Bounds check
                if self.page_counter >= len(self.main_df):
                     self.page_counter = len(self.main_df) - 1
                if self.page_counter < 0:
                    self.page_counter = 0
                if self.main_df.empty:
                    return

                # Get image path with safety checks
                image_path = self.main_df.iloc[self.page_counter]['Image_Path']
                if pd.isna(image_path):
                    messagebox.showerror("Error", "Invalid image path in database")
                    return

                # Use get_full_path to resolve relative paths
                image_path = self.get_full_path(image_path)

                if not os.path.exists(image_path):
                    messagebox.showerror("Error", f"Image file not found: {image_path}")
                    return

                # Update the current image path and load it
                self.current_image_path = image_path
                self.image_handler.load_image(self.current_image_path)

                # Load the text and update the counter
                self.load_text()
                self.counter_update()

                # Re-highlight search terms
                if hasattr(self, 'find_replace'):
                    self.find_replace.highlight_text()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to navigate images: {str(e)}")
                self.error_logging(f"Navigation error: {str(e)}")

    def navigate_relevant(self, direction):
        """Navigate to the next/previous relevant or partially relevant document."""
        if self.main_df.empty:
            messagebox.showinfo("No Documents", "No documents loaded.")
            return

        if 'Relevance' not in self.main_df.columns:
            messagebox.showerror("Error", "Relevance data not found. Run 'Find Relevant Documents' first.")
            return

        target_relevance = ["Relevant", "Partially Relevant"]
        current_index = self.page_counter
        total_rows = len(self.main_df)

        # Filter for relevant rows and get their indices
        relevant_indices = self.main_df[self.main_df['Relevance'].isin(target_relevance)].index.tolist()

        if not relevant_indices:
            messagebox.showinfo("Not Found", "No documents marked as Relevant or Partially Relevant.")
            return

        next_index = -1

        if direction == 1: # Forward
            # Find the first relevant index after the current one
            found_after = [idx for idx in relevant_indices if idx > current_index]
            if found_after:
                next_index = found_after[0]
            else:
                # Wrap around: find the first relevant index from the start
                next_index = relevant_indices[0] # Go to the first relevant item


        elif direction == -1: # Backward
            # Find the first relevant index before the current one
            found_before = [idx for idx in relevant_indices if idx < current_index]
            if found_before:
                next_index = found_before[-1]
            else:
                 # Wrap around: find the last relevant index from the end
                next_index = relevant_indices[-1] # Go to the last relevant item


        if next_index != -1 and next_index != current_index:
            # Save current state before navigating (check page_counter validity)
            if self.page_counter < len(self.main_df):
                current_display = self.main_df.loc[self.page_counter, 'Text_Toggle']
                if current_display != "None":
                    text = self.data_operations.clean_text(self.text_display.get("1.0", tk.END))
                    if current_display in ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]:
                        self.main_df.loc[self.page_counter, current_display] = text

                if hasattr(self, 'relevance_var') and 'Relevance' in self.main_df.columns:
                     self.main_df.loc[self.page_counter, 'Relevance'] = self.relevance_var.get()

            # Navigate
            self.page_counter = next_index
            self.refresh_display() # Use refresh_display for consistency
        elif next_index == current_index:
             messagebox.showinfo("Navigation Info", "Already at the only relevant document.")
        else:
            # Should only happen if relevant_indices was initially empty
            messagebox.showinfo("Not Found", "No other relevant documents found.")

    def document_page_nav(self, direction):
        """Navigate between images within the current document's list."""
        # Check if we have a list of images for the current document
        if not isinstance(self.current_image_path_list, list) or len(self.current_image_path_list) <= 1:
            # No list or only one image, nothing to navigate
            return

        total_doc_pages = len(self.current_image_path_list)

        # Calculate new index based on direction
        new_doc_index = self.current_doc_page_index
        if abs(direction) == 2: # Double arrow (<< or >>)
            if direction < 0: new_doc_index = 0 # Go to first page
            else: new_doc_index = total_doc_pages - 1 # Go to last page
        else: # Single arrow (< or >)
            new_doc_index += direction

        # Clamp index within bounds [0, total_doc_pages - 1]
        if new_doc_index < 0:
            new_doc_index = 0
        elif new_doc_index >= total_doc_pages:
            new_doc_index = total_doc_pages - 1

        # If the index actually changed, update the display
        if new_doc_index != self.current_doc_page_index:
            self.current_doc_page_index = new_doc_index

            try:
                # Get the new image path from the list
                image_path_to_display = str(self.current_image_path_list[self.current_doc_page_index]) # Ensure string
                image_path_abs = self.get_full_path(image_path_to_display)

                # Load the new image
                if image_path_abs and os.path.exists(image_path_abs):
                    self.current_image_path = image_path_abs
                    self.image_handler.load_image(self.current_image_path)

                    # Update the document page counter display
                    current_doc_page_num = self.current_doc_page_index + 1
                    self.doc_page_counter_var.set(f"{current_doc_page_num} / {total_doc_pages}")

                else:
                    messagebox.showerror("Error", f"Image file not found: {image_path_abs or image_path_to_display}")
                    self.image_display.delete("all")
                    self.current_image_path = None
                    # Update counter even if image fails to load
                    current_doc_page_num = self.current_doc_page_index + 1
                    self.doc_page_counter_var.set(f"{current_doc_page_num} / {total_doc_pages}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to navigate document page: {str(e)}")
                self.error_logging(f"Document page navigation error: {str(e)}")
        # No need to reload text, as text corresponds to the main document (page_counter), not the sub-image

# Image Functions

    def process_new_images(self, source_paths):
        try:
            # Call the ImageHandler method
            successful_copies, new_rows_list = self.image_handler.process_new_images(
                source_paths, 
                self.images_directory, 
                self.project_directory, 
                self.temp_directory, 
                self.main_df,
                lambda idx: setattr(self, 'page_counter', idx)
            )
            
            if successful_copies > 0:
                # Create a DataFrame from the list of dictionaries
                new_rows_df = pd.DataFrame(new_rows_list)
                # Concatenate with the main DataFrame
                self.main_df = pd.concat([self.main_df, new_rows_df], ignore_index=True)

                # Set text display to "None" before refreshing the display
                self.text_display_var.set("None")
                # Navigate to the first of the newly added images if appropriate
                if len(self.main_df) == successful_copies:  # If this was the first import
                    self.page_counter = 0
                else:
                    # Navigate to the first new image
                    self.page_counter = len(self.main_df) - successful_copies

                self.refresh_display()

                # Add auto-rotation if enabled in settings using the handler
                if hasattr(self, 'settings') and getattr(self.settings, 'check_orientation', False):
                    # First rotation pass
                    self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Auto_Rotate")

                    # Brief pause to ensure all rotations are complete
                    self.after(1000)  # 1 second pause

                    # Second rotation pass
                    self.ai_functions_handler.ai_function(all_or_one_flag="All Pages", ai_job="Auto_Rotate")
            else:
                messagebox.showinfo("Information", "No images were successfully processed")
        except Exception as e:
            self.error_logging(f"Error in process_new_images: {e}")
            messagebox.showerror("Error", f"Failed to process images: {e}")        

    def rotate_image(self, direction):
        if self.main_df.empty or self.page_counter >= len(self.main_df):
             return # Nothing to rotate
        image_path_rel = self.main_df.loc[self.page_counter, 'Image_Path']
        image_path_abs = self.get_full_path(image_path_rel)
        # --- EDIT: Pass angle directly --- 
        success, error_message = self.image_handler.rotate_image(image_path_abs, direction)
        # --- END EDIT ---
        if not success:
            messagebox.showerror("Error", error_message)

# Project Save and Open Handlers

    def import_pdf(self):
        """Opens a file dialog to select a PDF file and imports it"""
        pdf_file = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if pdf_file:
            self.open_pdf(pdf_file)

# Loading Functions

    def open_folder(self, toggle):
        directory = filedialog.askdirectory()
        if directory:
            # Use an absolute path for the project directory
            self.project_directory = os.path.abspath(directory)
            # Make sure images directory is relative to project or temp
            self.images_directory = os.path.join(self.project_directory, "images")
            os.makedirs(self.images_directory, exist_ok=True)

            # Reset application state.
            self.reset_application() # This resets main_df and page_counter

            if toggle == "Images without Text":
                self.load_files_from_folder_no_text()
            else:
                self.load_files_from_folder()
            self.enable_drag_and_drop()

    def load_files_from_folder(self):
        if not self.project_directory: # Check project_directory now
            messagebox.showerror("Error", "No directory selected.")
            return

        # Get lists of image and text files from the project_directory.
        all_image_files = [f for f in os.listdir(self.project_directory) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"))]
        text_files = [f for f in os.listdir(self.project_directory) if f.lower().endswith(".txt")]

        if not all_image_files:
            messagebox.showinfo("No Files", "No image files found in the selected directory.")
            return

        # Convert non-JPEG images to JPEG
        image_files = []
        original_files_to_move = []  # Track files that need to be moved after conversion

        for img_file in all_image_files:
            lower_file = img_file.lower()
            if lower_file.endswith((".jpg", ".jpeg")):
                image_files.append(img_file)
            elif lower_file.endswith((".png", ".tif", ".tiff", ".jp2")):
                # Convert to JPEG
                src_path = os.path.join(self.project_directory, img_file)
                base_name = os.path.splitext(img_file)[0]
                jpeg_name = base_name + '.jpg'
                dest_path = os.path.join(self.project_directory, jpeg_name)

                try:
                    img = Image.open(src_path)
                    # Handle transparency
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        alpha_mask = img.split()[-1] if len(img.split()) > 3 else None
                        background.paste(img, mask=alpha_mask)
                        img = background
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    img.save(dest_path, 'JPEG', quality=95)
                    image_files.append(jpeg_name)
                    original_files_to_move.append(img_file)  # Mark for moving
                except Exception as e:
                    self.error_logging(f"Error converting {img_file} to JPEG", f"{e}")
                    messagebox.showwarning("Conversion Error", f"Failed to convert {img_file}: {e}")

        # Move original non-JPEG files to subdirectory
        if original_files_to_move:
            original_dir = os.path.join(self.project_directory, 'original_non-jpeg_images')
            os.makedirs(original_dir, exist_ok=True)

            for orig_file in original_files_to_move:
                src = os.path.join(self.project_directory, orig_file)
                dst = os.path.join(original_dir, orig_file)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    self.error_logging(f"Error moving {orig_file} to subdirectory", f"{e}")

        if not image_files:
            messagebox.showinfo("No Files", "No valid image files could be loaded.")
            return

        # Create a dictionary of text files for easy lookup by name (without extension)
        text_files_dict = {}
        for text_file in text_files:
            text_base_name = os.path.splitext(text_file)[0]
            text_files_dict[text_base_name] = text_file

        # Sort image files naturally (handling both numeric and text-based filenames)
        def natural_sort_key(s):
            # Split the string into text and numeric parts
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'([0-9]+)', s)]

        image_files.sort(key=natural_sort_key)

        new_rows_list = []
        # Populate the DataFrame with all image files
        for i, image_file in enumerate(image_files):
            image_path_abs = os.path.join(self.project_directory, image_file)
            image_path_rel = self.get_relative_path(image_path_abs) # Store relative path

            # Get the image base name (without extension)
            image_base_name = os.path.splitext(image_file)[0]

            # Check if there's a matching text file
            text_content = ""
            text_path_rel = ""
            if image_base_name in text_files_dict:
                matching_text_file = text_files_dict[image_base_name]
                text_path_abs = os.path.join(self.project_directory, matching_text_file)
                text_path_rel = self.get_relative_path(text_path_abs) # Store relative path

                # Read text content using UTF-8 encoding
                try:
                    with open(text_path_abs, "r", encoding='utf-8') as f:
                        text_content = f.read()
                except Exception as e:
                    self.error_logging(f"Error reading text file {text_path_abs}", f"{e}")

            # Set Text_Toggle to "Original_Text" only if text_content is non-empty; otherwise "None"
            text_toggle = "Original_Text" if text_content.strip() else "None"

            page = f"{i+1:04d}_p{i+1:03d}"
            new_row_data = {
                "Index": i,
                "Page": page,
                "Original_Text": text_content,
                "Corrected_Text": "",
                "Formatted_Text": "",
                "Translation": "",
                "Image_Path": image_path_rel, # Use relative path
                "Text_Path": text_path_rel,   # Use relative path
                "Text_Toggle": text_toggle,
                "People": "",
                "Places": "",
                "Errors": "",
                "Errors_Source": "",
                "Relevance": ""
            }
            # Ensure all DataFrame columns are present
            for col in self.main_df.columns:
                if col not in new_row_data:
                    new_row_data[col] = ""

            new_rows_list.append(new_row_data)

        # Create DataFrame from list
        self.main_df = pd.DataFrame(new_rows_list)

        # Load the first image and its text.
        if len(self.main_df) > 0:
            self.page_counter = 0 # Reset counter after loading
            self.current_image_path = self.get_full_path(self.main_df.loc[0, 'Image_Path'])
            if self.current_image_path and os.path.exists(self.current_image_path):
                 self.image_handler.load_image(self.current_image_path)
            else:
                 messagebox.showerror("Error", f"Image not found: {self.current_image_path}")


            # Ensure text_display_var is set to the proper value based on text_toggle
            if self.main_df.loc[0, 'Text_Toggle'] == "None":
                self.text_display_var.set("None")
            else:
                self.text_display_var.set(self.main_df.loc[0, 'Text_Toggle'])

            self.load_text()
        else:
            messagebox.showinfo("No Files", "No files found in the selected directory.")

        self.counter_update()

    def load_files_from_folder_no_text(self):
        if not self.project_directory: # Check project_directory
             messagebox.showerror("Error", "No directory selected.")
             return
        # Reset the page counter (already done in open_folder)
        # self.page_counter = 0

        # Load image files from project_directory
        all_image_files = [file for file in os.listdir(self.project_directory)
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"))]

        if not all_image_files:
            messagebox.showinfo("No Files", "No image files found in the selected directory.")
            return

        # Convert non-JPEG images to JPEG
        image_files = []
        original_files_to_move = []  # Track files that need to be moved after conversion

        for img_file in all_image_files:
            lower_file = img_file.lower()
            if lower_file.endswith((".jpg", ".jpeg")):
                image_files.append(img_file)
            elif lower_file.endswith((".png", ".tif", ".tiff", ".jp2")):
                # Convert to JPEG
                src_path = os.path.join(self.project_directory, img_file)
                base_name = os.path.splitext(img_file)[0]
                jpeg_name = base_name + '.jpg'
                dest_path = os.path.join(self.project_directory, jpeg_name)

                try:
                    img = Image.open(src_path)
                    # Handle transparency
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        alpha_mask = img.split()[-1] if len(img.split()) > 3 else None
                        background.paste(img, mask=alpha_mask)
                        img = background
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    img.save(dest_path, 'JPEG', quality=95)
                    image_files.append(jpeg_name)
                    original_files_to_move.append(img_file)  # Mark for moving
                except Exception as e:
                    self.error_logging(f"Error converting {img_file} to JPEG", f"{e}")
                    messagebox.showwarning("Conversion Error", f"Failed to convert {img_file}: {e}")

        # Move original non-JPEG files to subdirectory
        if original_files_to_move:
            original_dir = os.path.join(self.project_directory, 'original_non-jpeg_images')
            os.makedirs(original_dir, exist_ok=True)

            for orig_file in original_files_to_move:
                src = os.path.join(self.project_directory, orig_file)
                dst = os.path.join(original_dir, orig_file)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    self.error_logging(f"Error moving {orig_file} to subdirectory", f"{e}")

        if not image_files:
            messagebox.showinfo("No Files", "No valid image files could be loaded.")
            return

        # Sort image files naturally (handling both numeric and text-based filenames)
        def natural_sort_key(s):
            # Split the string into text and numeric parts
            return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'([0-9]+)', s)]

        image_files.sort(key=natural_sort_key)

        new_rows_list = []
        # Populate the DataFrame
        for i, image_file in enumerate(image_files):
            image_path_abs = os.path.join(self.project_directory, image_file)
            image_path_rel = self.get_relative_path(image_path_abs) # Store relative

            # No text file needed for this import mode
            text_path_rel = ""

            page = f"{i+1:04d}_p{i+1:03d}"
            new_row_data = {
                "Index": i,
                "Page": page,
                "Original_Text": "",
                "Corrected_Text": "",
                "Formatted_Text": "",
                "Translation": "",
                "Image_Path": image_path_rel, # Relative path
                "Text_Path": text_path_rel,   # Relative path (empty)
                "Text_Toggle": "None",
                "People": "",
                "Places": "",
                "Errors": "",
                "Errors_Source": "",
                "Relevance": ""
            }
             # Ensure all DataFrame columns are present
            for col in self.main_df.columns:
                if col not in new_row_data:
                    new_row_data[col] = ""

            new_rows_list.append(new_row_data)

        # Create DataFrame
        self.main_df = pd.DataFrame(new_rows_list)

        # Load the first image and text file
        if len(self.main_df) > 0:
            self.page_counter = 0 # Reset counter
            self.current_image_path = self.get_full_path(self.main_df.loc[0, 'Image_Path'])
            if self.current_image_path and os.path.exists(self.current_image_path):
                self.image_handler.load_image(self.current_image_path)
            else:
                 messagebox.showerror("Error", f"Image not found: {self.current_image_path}")
            self.text_display_var.set("None")  # Set dropdown to "None"
            self.load_text()
        else:
            messagebox.showinfo("No Files", "No files found in the selected directory.")

        self.counter_update()

    def load_text(self):
        # Prevent updates if metadata generation is happening with a temporary df
        if getattr(self, '_is_generating_export_metadata', False):
            self.error_logging("Skipped load_text due to _is_generating_export_metadata flag.", level="INFO")
            return

        # Ensure there is at least one row and that the page counter is valid.
        index = self.page_counter
        row_data = self.main_df.loc[index]
        current_toggle = row_data.get('Text_Toggle', "None")

        # Set dropdown to current toggle
        display_map = {
            "None": "None",
            "Original_Text": "Original_Text",
            "Corrected_Text": "Corrected_Text",
            "Formatted_Text": "Formatted_Text",
            "Translation": "Translation",
            "Separated_Text": "Separated_Text"
        }
        self.text_display_var.set(display_map.get(current_toggle, "None"))

        # Based on the toggle, select the text to display. Use .get() for safety.
        text_column = display_map.get(current_toggle)
        if text_column and text_column != "None" and text_column in row_data:
            text = row_data.get(text_column, "") # Get text, default to "" if column missing
            text = text if pd.notna(text) else "" # Ensure it's not NaN
        else:
            text = "" # Default for "None" or missing column

        self.text_display.delete("1.0", tk.END)
        if text:
            self.text_display.insert("1.0", text)

        # Update the dropdown options dynamically based on available text.
        available_options = ["None"]
        if pd.notna(row_data.get('Original_Text')) and row_data.get('Original_Text', "").strip():
            available_options.append("Original_Text")
        if pd.notna(row_data.get('Corrected_Text')) and row_data.get('Corrected_Text', "").strip():
            available_options.append("Corrected_Text")
        if pd.notna(row_data.get('Formatted_Text')) and row_data.get('Formatted_Text', "").strip():
            available_options.append("Formatted_Text")
        if pd.notna(row_data.get('Translation')) and row_data.get('Translation', "").strip():
            available_options.append("Translation")
        if pd.notna(row_data.get('Separated_Text')) and row_data.get('Separated_Text', "").strip():
            available_options.append("Separated_Text")
        self.text_display_dropdown['values'] = available_options

        # Load and set the relevance value for the current page
        if 'Relevance' in self.main_df.columns:
            relevance_value = row_data.get('Relevance', "") # Default to ""
            # Set to empty string if NaN or None
            if pd.isna(relevance_value) or relevance_value is None:
                relevance_value = ""

            # Update relevance dropdown
            try:
                self.relevance_var.set(relevance_value)
                # If we have a non-empty relevance value, make sure the dropdown is visible
                if relevance_value.strip() and not self.show_relevance.get():
                    self.show_relevance.set(True)
                    self.toggle_relevance_visibility()
            except Exception as e:
                self.error_logging(f"Error setting relevance value: {str(e)}")
                self.relevance_var.set("")
        else:
            # Ensure dropdown is cleared if column doesn't exist
            self.relevance_var.set("")

        # Re-highlight find/replace matches if that window is active
        if hasattr(self, 'find_replace') and self.find_replace.find_replace_toggle:
            self.find_replace.highlight_text()

        # Apply any active highlighting (names, places, changes, errors)
        self.highlight_handler.highlight_text() # <-- Use handler

        # Update menu states based on data availability
        self.update_highlight_menu_states()

        self.counter_update()

# Utility Functions

    def undo(self, event=None):
        try:
            self.text_display.edit_undo()
        except tk.TclError:
            pass

    def redo(self, event=None):
        try:
            self.text_display.edit_redo()
        except tk.TclError:
            pass

    def increase_font_size(self):
        """Increase the font size of the text display"""
        if self.text_font_size < 48:  # Maximum font size limit
            self.text_font_size += 2
            self.update_text_font()

    def decrease_font_size(self):
        """Decrease the font size of the text display"""
        if self.text_font_size > 8:  # Minimum font size limit
            self.text_font_size -= 2
            self.update_text_font()

    def update_text_font(self):
        """Update the font size of the text display widget"""
        self.text_display.config(font=("Calibri", self.text_font_size))

    def update_recent_projects_menu(self):
        """Update the recent projects submenu with current recent projects"""
        # Clear existing menu items
        self.recent_menu.delete(0, 'end')
        
        # Get recent projects from settings
        recent_projects = self.settings.get_recent_projects()
        
        if not recent_projects:
            # Add a disabled "No recent projects" item if list is empty
            self.recent_menu.add_command(label="No recent projects", state="disabled")
        else:
            # Add menu items for each recent project
            for i, project_path in enumerate(recent_projects):
                # Get just the project name (directory name) for display
                project_name = os.path.basename(project_path)
                if not project_name:  # Handle case where path ends with separator
                    project_name = os.path.basename(os.path.dirname(project_path))
                
                # Add menu item with keyboard shortcut for first 4 items
                if i < 9:  # Support up to 9 recent projects with number shortcuts
                    label = f"&{i+1} {project_name}"
                else:
                    label = project_name
                
                self.recent_menu.add_command(
                    label=label,
                    command=lambda path=project_path: self.open_recent_project(path)
                )
            
            # Add separator and clear menu option
            if recent_projects:
                self.recent_menu.add_separator()
                self.recent_menu.add_command(
                    label="Clear Recent Projects",
                    command=self.clear_recent_projects
                )

    def open_recent_project(self, project_path):
        """Open a project from the recent projects list"""
        try:
            # Check if the project still exists
            if not os.path.exists(project_path):
                messagebox.showerror("Error", f"Project not found: {project_path}")
                # Remove the non-existent project from recent list
                if project_path in self.settings.recent_projects:
                    self.settings.recent_projects.remove(project_path)
                    self.settings.save_settings()
                    self.update_recent_projects_menu()
                return
            
            # Save current text before opening new project
            self.data_operations.update_df()
            
            # Open the project using ProjectIO
            project_name = os.path.basename(project_path)
            project_file = os.path.join(project_path, f"{project_name}.pbf")
            images_directory = os.path.join(project_path, "images")

            if not os.path.exists(project_file) or not os.path.exists(images_directory):
                messagebox.showerror("Error", "Invalid project directory. Missing project file or images directory.")
                return

            # Use the existing project opening logic from ProjectIO
            # Set the project path and call the open project method
            original_method = self.project_io.open_project
            
            # Temporarily override the directory selection to use our path
            def mock_askdirectory(*args, **kwargs):
                return project_path
            
            # Backup the original filedialog.askdirectory
            original_askdirectory = filedialog.askdirectory
            filedialog.askdirectory = mock_askdirectory
            
            try:
                # Call the regular open project method
                self.project_io.open_project()
                
                # Add this project to recent projects (moves it to top)
                self.settings.add_recent_project(project_path)
                self.update_recent_projects_menu()
                
            finally:
                # Restore the original filedialog.askdirectory
                filedialog.askdirectory = original_askdirectory
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open recent project: {e}")
            self.error_logging(f"Failed to open recent project {project_path}: {e}")

    def clear_recent_projects(self):
        """Clear all recent projects from the list"""
        if messagebox.askyesno("Clear Recent Projects", "Are you sure you want to clear all recent projects?"):
            self.settings.recent_projects = []
            self.settings.save_settings()
            self.update_recent_projects_menu()

    def error_logging(self, error_message, additional_info=None, level="ERROR"):
        """Use the ErrorLogger module to log errors."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_level_setting = getattr(self.settings, 'log_level', 'ERROR')
        log_error(base_dir, log_level_setting, error_message, additional_info, level)

    def drop(self, event):
        file_paths = event.data
        # Improved parsing for paths with spaces, potentially in braces
        try:
             # Try literal evaluation if it looks like a list/tuple
             if file_paths.startswith( ('(', '[', '{') ) and file_paths.endswith( (')', ']', '}') ):
                 evaluated_paths = ast.literal_eval(file_paths)
                 if isinstance(evaluated_paths, (list, tuple)):
                     file_paths = [str(p) for p in evaluated_paths]
                 else: # If it evaluates to something else, fall back to regex
                     file_paths = re.findall(r'\{.*?\}|\S+', event.data)
             else: # Standard regex for space-separated or brace-enclosed
                 file_paths = re.findall(r'\{.*?\}|\S+', event.data)
        except:
            # Fallback if literal_eval fails
            file_paths = re.findall(r'\{.*?\}|\S+', event.data)


        valid_images = []
        pdf_files = []
        invalid_files = []

        # Record current image count before processing new files
        prev_count = len(self.main_df)

        # Process all files first
        for file_path in file_paths:
            # Remove curly braces and any quotation marks
            file_path = file_path.strip('{}').strip('"').strip()

            if os.path.isfile(file_path):
                lower_path = file_path.lower()
                if lower_path.endswith(('.jpg', '.jpeg')):
                    valid_images.append(file_path)
                elif lower_path.endswith(('.png', '.tif', '.tiff', '.jp2')):
                    try:
                        img = Image.open(file_path)
                        # Ensure RGBA or LA images have a white background when converted
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            # Paste using alpha channel as mask
                            alpha_mask = img.split()[-1] if len(img.split()) > 3 else None
                            background.paste(img, mask=alpha_mask)
                            img = background
                        # Convert to RGB if not already
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Define path for converted JPEG in edit_temp
                        jpeg_filename = os.path.splitext(os.path.basename(file_path))[0] + '_converted.jpg'
                        jpeg_path = os.path.join(self.edit_temp_directory, jpeg_filename) # Use edit_temp

                        img.save(jpeg_path, 'JPEG', quality=95)
                        valid_images.append(jpeg_path) # Add the path of the *converted* image
                    except Exception as e:
                        self.error_logging(f"Error converting image file {file_path}", f"{e}")
                        invalid_files.append(file_path)
                elif lower_path.endswith('.pdf'):
                    pdf_files.append(file_path)
                else:
                    invalid_files.append(file_path)
            elif os.path.isdir(file_path): # Handle dropped folders
                 messagebox.showinfo("Folder Dropped", f"Folder '{os.path.basename(file_path)}' dropped. Please use File > Import menu to import folders.")
            else:
                invalid_files.append(file_path)

        # Process valid image files
        if valid_images:
            self.process_new_images(valid_images)


            # Clean up any temporary converted PNG->JPG files from edit_temp
            for image_path in valid_images:
                if image_path.endswith('_converted.jpg') and os.path.dirname(image_path) == self.edit_temp_directory:
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                    except Exception as e:
                        self.error_logging(f"Error removing temporary file {image_path}", f"{e}")

        # Process PDF files
        if pdf_files:
            for pdf_file in pdf_files:
                try:
                    self.project_io.open_pdf(pdf_file) # <--- Corrected call
                except Exception as e:
                    self.error_logging(f"Error processing PDF file {pdf_file}", f"{e}")
                    messagebox.showerror("Error", f"Failed to process PDF file {pdf_file}: {e}")

        # Report invalid files
        if invalid_files:
            invalid_files_str = "\n".join(invalid_files)
            self.error_logging(f"Invalid files/paths not processed: {invalid_files_str}", level="WARNING")
            messagebox.showwarning("Invalid Files",
                f"The following files/paths were not processed because they are not valid image or PDF files:\n\n{invalid_files_str}")

    def get_full_path(self, path):
        """ Resolves a potentially relative path to an absolute path based on the project directory. """
        # If the path isn't a string or is empty, return it as is.
        if not isinstance(path, str) or not path.strip():
            return path
        # If it's already absolute, return it.
        if os.path.isabs(path):
            return path
        # If a project is open, join with the project directory.
        # Use self.project_directory if it exists, otherwise use temp_directory as a fallback base
        base_dir = getattr(self, 'project_directory', None) or getattr(self, 'temp_directory', None)
        if base_dir:
            # Normalize both base_dir and path to handle mixed slashes before joining
            normalized_base = os.path.normpath(base_dir)
            normalized_path = os.path.normpath(path)
            full_path = os.path.join(normalized_base, normalized_path)
            return os.path.abspath(full_path) # Ensure it's truly absolute

        # Fallback: return the absolute path relative to the current working directory (less ideal)
        return os.path.abspath(os.path.normpath(path))

    def get_relative_path(self, path, base_path=None):
        """
        Convert an absolute path to a path relative to base_path.
        If base_path is not provided, use self.project_directory.
        """
        # If the path isn't a string or is empty, return an empty string
        if not isinstance(path, str) or not path:
            return ""

        # Ensure path is absolute before attempting to make it relative
        abs_path = os.path.abspath(path)

        # Use project_directory as base_path if not provided
        if base_path is None:
            base_path = getattr(self, 'project_directory', None) or getattr(self, 'temp_directory', None)

        if not base_path:
             # No base path to make relative to, return the absolute path
             return abs_path

        # Ensure base_path is absolute
        abs_base_path = os.path.abspath(base_path)

        try:
            # Convert absolute path to path relative to base_path
            # Normalize paths first to handle potential inconsistencies (e.g., drive letters)
            rel_path = os.path.relpath(os.path.normpath(abs_path), os.path.normpath(abs_base_path))
            return rel_path
        except ValueError:
            # Handle case where paths are on different drives in Windows
            # In this case, we cannot create a relative path, so return the absolute path.
            return abs_path

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Save any pending changes before quitting using DataOperations
            self.data_operations.update_df()

            self.quit()
            self.destroy() # Ensure window closes fully

# GUI Actions / Toggles

    def toggle_button_state(self):

        if self.button1['state'] == "normal" and self.button2['state'] == "normal" and self.button4['state'] == "normal" and self.button5['state'] == "normal":
            self.button1.config(state="disabled")
            self.button2.config(state="disabled")
            self.button4.config(state="disabled")
            self.button5.config(state="disabled")

        else:
            self.button1.config(state="normal")
            self.button2.config(state="normal")
            self.button4.config(state="normal")
            self.button5.config(state="normal")

    def toggle_text(self):
        if self.main_df.empty or self.page_counter >= len(self.main_df):
            return

        # Get current state before toggling
        index = self.page_counter
        row_data = self.main_df.loc[index]
        current_toggle = row_data.get('Text_Toggle', "None")

        # Only save changes if we're not in "None" mode
        if current_toggle != "None":
            # Get the current text from the text widget
            text = self.data_operations.clean_text(self.text_display.get("1.0", tk.END))

            # Save the text to the appropriate column based on CURRENT display type
            if current_toggle in ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]:
                self.main_df.loc[index, current_toggle] = text

        # Determine available text types
        has_separated = pd.notna(row_data.get('Separated_Text')) and row_data.get('Separated_Text', "").strip()
        has_translation = pd.notna(row_data.get('Translation')) and row_data.get('Translation', "").strip()
        has_formatted = pd.notna(row_data.get('Formatted_Text')) and row_data.get('Formatted_Text', "").strip()
        has_corrected = pd.notna(row_data.get('Corrected_Text')) and row_data.get('Corrected_Text', "").strip()
        has_original = pd.notna(row_data.get('Original_Text')) and row_data.get('Original_Text', "").strip()

        # Define the toggle order
        toggle_order = ["Original_Text", "Corrected_Text", "Formatted_Text", "Translation", "Separated_Text"]
        available_toggles = [t for t in toggle_order if locals()[f'has_{t.split("_")[0].lower()}']]

        if not available_toggles: # If no text available at all
             self.main_df.loc[index, 'Text_Toggle'] = "None"
             self.load_text()
             return

        # Find the index of the current toggle in the available list
        try:
            current_idx = available_toggles.index(current_toggle)
        except ValueError:
            # If current toggle isn't available (e.g., was deleted), start from the first available
            current_idx = -1

        # Calculate the next index, wrapping around
        next_idx = (current_idx + 1) % len(available_toggles)
        next_toggle = available_toggles[next_idx]

        self.main_df.loc[index, 'Text_Toggle'] = next_toggle
        self.load_text()

        # Apply highlighting based on toggle states without mutual exclusivity

        # Apply highlighting
        self.highlight_handler.highlight_text() # <-- Use handler

        # Update menu item states
        self.update_highlight_menu_states()

    def toggle_relevance_display(self, event=None):
         """Toggle the visibility of the entire relevance section (dropdown and buttons)"""
         self.show_relevance.set(not self.show_relevance.get())
         self.toggle_relevance_visibility() # This now handles dropdown and buttons together

    def toggle_nav_display(self, event=None):
        """Toggle the visibility of the document navigation controls"""
        self.show_page_nav.set(not self.show_page_nav.get())
        self.toggle_page_nav_visibility()

    def toggle_highlight_options(self):
        """Update highlight display based on toggle states without mutual exclusivity"""

        # Apply highlighting
        self.highlight_handler.highlight_text() # <-- Use handler

        # Update menu item states
        self.update_highlight_menu_states()

    def toggle_relevance_visibility(self):
        """Toggles the visibility of the relevance dropdown, label, AND navigation buttons."""
        if self.show_relevance.get():
            # Show relevance elements
            self.relevance_label.pack(side="left", padx=2)
            self.relevance_dropdown.pack(side="left", padx=2)
            # Show navigation buttons directly
            self.relevant_back_button.pack(side="left", padx=(5, 2))
            self.relevant_forward_button.pack(side="left", padx=2)
        else:
            # Hide relevance elements
            self.relevance_label.pack_forget()
            self.relevance_dropdown.pack_forget()
            # Hide navigation buttons directly
            self.relevant_back_button.pack_forget()
            self.relevant_forward_button.pack_forget()

    def toggle_page_nav_visibility(self):
        """Toggles the visibility of the document page navigation controls."""
        if self.show_page_nav.get():
            self.doc_page_label.pack(side="left", padx=(5, 2))
            self.doc_button1.pack(side="left", padx=2)
            self.doc_button2.pack(side="left", padx=2)
            self.doc_page_counter_label.pack(side="left", padx=2)
            self.doc_button4.pack(side="left", padx=2)
            self.doc_button5.pack(side="left", padx=2)
        else:
            self.doc_page_label.pack_forget()
            self.doc_button1.pack_forget()
            self.doc_button2.pack_forget()
            self.doc_page_counter_label.pack_forget()
            self.doc_button4.pack_forget()
            self.doc_button5.pack_forget()

# Update Functions

    def refresh_display(self):
        """Refresh the current image and text display, handling single and multi-image paths."""
        if self.main_df.empty:
            self.error_logging("Refresh display called with empty DataFrame", level="INFO")
            self.image_display.delete("all")
            self.text_display.delete("1.0", tk.END)
            self.text_display_var.set("None")
            self.current_image_path_list = None # Reset list state
            self.current_doc_page_index = 0
            self.show_page_nav.set(False) # Hide doc nav
            self.toggle_page_nav_visibility()
            self.counter_update()
            return

        # Ensure page_counter is within valid bounds
        if self.page_counter >= len(self.main_df): self.page_counter = len(self.main_df) - 1
        if self.page_counter < 0: self.page_counter = 0

        # Reset state for the new page/document
        self.current_image_path_list = None
        image_path_to_display = None # The single path string we'll use

        try:
            # Get image path data from DataFrame
            image_path_data = self.main_df.iloc[self.page_counter]['Image_Path']

            # --- Check if Image_Path is a list or single path ---
            if isinstance(image_path_data, list):
                if not image_path_data: # Empty list
                    self.error_logging(f"Error: Image_Path list is empty at index {self.page_counter}.", level="ERROR")
                    image_path_to_display = None
                    self.current_image_path_list = [] # Store empty list
                else:
                    self.current_image_path_list = image_path_data # Store the list
                    # Ensure current_doc_page_index is valid for this list
                    if self.current_doc_page_index < 0 or self.current_doc_page_index >= len(self.current_image_path_list):
                        self.current_doc_page_index = 0 # Reset if out of bounds
                    # Get the specific path to display based on index
                    image_path_to_display = str(self.current_image_path_list[self.current_doc_page_index]) # Ensure string

            elif isinstance(image_path_data, str):
                image_path_to_display = image_path_data # It's a single path string
                self.current_image_path_list = None # Not a list
                self.current_doc_page_index = 0
            elif pd.isna(image_path_data):
                image_path_to_display = None # Handle None or NaN
                self.current_image_path_list = None
                self.current_doc_page_index = 0
            else:
                # Handle other unexpected types
                self.error_logging(f"Warning: Unexpected type for Image_Path at index {self.page_counter}: {type(image_path_data)}. Treating as empty.", level="WARNING")
                image_path_to_display = None
                self.current_image_path_list = None
                self.current_doc_page_index = 0
            # --- End Path Type Check ---

            # Convert the selected path to absolute path if it exists
            image_path_abs = self.get_full_path(image_path_to_display) if image_path_to_display else ""

            # --- Load Image ---
            if image_path_abs and os.path.exists(image_path_abs):
                self.current_image_path = image_path_abs # Keep track of the currently displayed absolute path
                self.image_handler.load_image(self.current_image_path)
            else:
                if image_path_to_display: # Only show error if there was supposed to be a path
                    messagebox.showerror("Error", f"Image file not found: {image_path_abs or image_path_to_display}")
                self.image_display.delete("all") # Clear image if not found or path was bad
                self.current_image_path = None
            # --- End Load Image --
            # --- Update Document Page Navigation Visibility and Counter ---
            if isinstance(self.current_image_path_list, list) and len(self.current_image_path_list) > 1:
                self.show_page_nav.set(True)
                total_doc_pages = len(self.current_image_path_list)
                current_doc_page_num = self.current_doc_page_index + 1
                self.doc_page_counter_var.set(f"{current_doc_page_num} / {total_doc_pages}")
            else:
                self.show_page_nav.set(False)
                self.doc_page_counter_var.set("0 / 0") # Or "1 / 1" if desired for single images
            self.toggle_page_nav_visibility() # Apply visibility change
            # --- End Document Page Navigation Update ---

            # --- Load Text ---
            # Make sure the text_display_var matches the Text_Toggle
            current_toggle = self.main_df.loc[self.page_counter, 'Text_Toggle']
            self.text_display_var.set(current_toggle if pd.notna(current_toggle) else "None")
            self.load_text() # Load_text handles counter update and highlights
            # --- End Load Text ---

        except Exception as e:
            # Catch potential errors during path resolution or loading
            error_msg = f"Failed to refresh display: {str(e)}\nIndex: {self.page_counter}"
            if isinstance(e, TypeError) and "_path_exists" in str(e):
                 error_msg = f"Failed to refresh display: Image path type error.\nIndex: {self.page_counter}\nError: {e}"
            if isinstance(e, IndexError):
                error_msg += f"\nLikely invalid doc page index: {self.current_doc_page_index}"

            messagebox.showerror("Error", error_msg)
            self.error_logging(f"Refresh display error: {error_msg}", level="ERROR")

            # Attempt to clear displays gracefully on error
            self.image_display.delete("all")
            self.current_image_path = None
            self.current_image_path_list = None
            self.current_doc_page_index = 0
            self.show_page_nav.set(False)
            self.toggle_page_nav_visibility()
            self.load_text() # Try loading text anyway

    def on_text_display_change(self, event=None):
        if self.main_df.empty or self.page_counter >= len(self.main_df):
            return

        # Get new selection and update toggle
        index = self.page_counter
        selected = self.text_display_var.get()

        # Map display names to DataFrame values
        display_map = {
            "None": "None",
            "Original_Text": "Original_Text",
            "Corrected_Text": "Corrected_Text",
            "Formatted_Text": "Formatted_Text",
            "Translation": "Translation",
            "Separated_Text": "Separated_Text"
        }

        # Update the Text_Toggle in the DataFrame
        self.main_df.at[index, 'Text_Toggle'] = display_map[selected]

        # Clear error highlights if switching to a text version they don't apply to
        if 'Errors_Source' in self.main_df.columns:
            errors_source = self.main_df.loc[index, 'Errors_Source']
            if errors_source and errors_source != selected:
                # We're switching to a text version that doesn't match the errors
                self.text_display.tag_remove("error_highlight", "1.0", tk.END)
            elif errors_source == selected and self.highlight_errors_var.get():
                # We're switching to the text version that matches the errors
                # Make sure error highlighting is enabled if there are errors
                errors = self.main_df.loc[index, 'Errors'] if 'Errors' in self.main_df.columns else ""
                if pd.notna(errors) and errors.strip():
                    self.highlight_errors_var.set(True)

        # Reload the text
        self.load_text()

        # Apply highlighting based on current settings - called within load_text
        # self.highlight_text()

    def update_highlight_menu_states(self):
        """Enable or disable highlight menu items based on data availability"""
        if self.main_df.empty or self.page_counter < 0 or self.page_counter >= len(self.main_df):
            # Disable all highlight options if no data is loaded
            self.document_menu.entryconfig("Highlight Names", state="disabled")
            self.document_menu.entryconfig("Highlight Places", state="disabled")
            self.document_menu.entryconfig("Highlight Changes", state="disabled")
            self.document_menu.entryconfig("Highlight Errors", state="disabled")
            return

        # Get current page data
        index = self.page_counter
        row_data = self.main_df.loc[index]
        current_toggle = row_data.get('Text_Toggle', "None")

        # Check for names data
        names_data = row_data.get('People', "")
        has_names = pd.notna(names_data) and names_data.strip() != ""
        self.document_menu.entryconfig("Highlight Names", state="normal" if has_names else "disabled")

        # Check for places data
        places_data = row_data.get('Places', "")
        has_places = pd.notna(places_data) and places_data.strip() != ""
        self.document_menu.entryconfig("Highlight Places", state="normal" if has_places else "disabled")

        # Check for changes - needs at least two text versions to compare
        has_changes = False
        if current_toggle == "Corrected_Text":
            original_text = row_data.get('Original_Text', "")
            Corrected_Text = row_data.get('Corrected_Text', "")
            has_changes = (pd.notna(original_text) and original_text.strip() != "" and
                           pd.notna(Corrected_Text) and Corrected_Text.strip() != "")
        elif current_toggle == "Formatted_Text":
            Corrected_Text = row_data.get('Corrected_Text', "")
            Formatted_Text = row_data.get('Formatted_Text', "")
            # Compare Formatted to Corrected first
            has_changes = (pd.notna(Corrected_Text) and Corrected_Text.strip() != "" and
                           pd.notna(Formatted_Text) and Formatted_Text.strip() != "")
            # If Corrected is empty, compare Formatted to Original
            if not has_changes:
                original_text = row_data.get('Original_Text', "")
                has_changes = (pd.notna(original_text) and original_text.strip() != "" and
                               pd.notna(Formatted_Text) and Formatted_Text.strip() != "")
        elif current_toggle == "Translation":
             # Compare Translation to Corrected_Text or Original_Text
            translation_text = row_data.get('Translation', "")
            if pd.notna(translation_text) and translation_text.strip():
                Corrected_Text = row_data.get('Corrected_Text', "")
                original_text = row_data.get('Original_Text', "")
                # Check against Corrected first, then Original
                if pd.notna(Corrected_Text) and Corrected_Text.strip():
                     has_changes = True
                elif pd.notna(original_text) and original_text.strip():
                     has_changes = True
        # Could add more comparisons here (e.g., Separated vs Formatted)

        self.document_menu.entryconfig("Highlight Changes", state="normal" if has_changes else "disabled")

        # Check for errors data - must have data and be viewing the right text version
        has_errors = False
        current_display = self.text_display_var.get()
        if 'Errors' in self.main_df.columns and 'Errors_Source' in self.main_df.columns:
            errors = row_data.get('Errors', "")
            errors_source = row_data.get('Errors_Source', "")

            # Only count errors if they exist and apply to the current text version
            has_errors = (pd.notna(errors) and errors.strip() != "" and
                         (not errors_source or errors_source == current_display)) # Allow if source matches or is empty

        self.document_menu.entryconfig("Highlight Errors", state="normal" if has_errors else "disabled")

    def update_separation_menu_state(self, state="normal"):
        """Enable or disable the 'Apply Document Separation' menu item."""
        try:
            # Get the current index of the menu entry
            menu_index = self.process_menu.index("Apply Document Separation")
            # Check if the index is valid (not None)
            if menu_index is not None:
                 self.process_menu.entryconfig(menu_index, state=state)
                 print(f"Set 'Apply Document Separation' state to {state}")
            else:
                 print(f"Could not find menu item 'Apply Document Separation' to set state.")
        except tk.TclError as e:
            # Log the error if the menu item doesn't exist
            self.error_logging(f"Error configuring 'Apply Document Separation' menu item: {e}", level="WARNING")

# GUI and DF Update Functions

    def update_image_rotation(self, index, response):
        if self.main_df.empty or index >= len(self.main_df):
             return

        # Get the image path from the DataFrame (relative).
        image_path_rel = self.main_df.loc[index, 'Image_Path']
        # Use get_full_path to resolve to absolute path for modification.
        image_path_abs = self.get_full_path(image_path_rel)

        if not image_path_abs or not os.path.exists(image_path_abs):
             self.error_logging(f"Image path not found or invalid for rotation at index {index}: {image_path_abs}")
             return

        # --- Start Edit: Updated Parsing Logic ---
        # Dictionary mapping the *specific expected terms* from the prompt to rotation angles.
        # Angle represents the rotation needed to make it upright.
        orientation_map = {
            "top": 0,
            "left": 90,
            "right": -90, # or 270
            "bottom": 180
        }

        # Find the term after "Orientation:" (case-insensitive)
        try:
            # Split the response string by "Orientation:", ignoring case, taking the last part
            parts = re.split(r'Orientation:', response, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                orientation_term = parts[1].strip().lower()
                # Remove potential trailing punctuation
                orientation_term = re.sub(r'[.,!?;:]$', '', orientation_term).strip()

                # Find the correction angle based on the extracted term
                correction_angle = orientation_map.get(orientation_term) # Use .get for safer lookup

            else:
                # "Orientation:" not found in the response
                self.error_logging(f"Could not find 'Orientation:' marker in response at index {index}: {response}")
                correction_angle = None

        except Exception as parse_error:
            self.error_logging(f"Error parsing rotation response at index {index}: {response} | Error: {parse_error}")
            correction_angle = None

        # If no valid term found or parsing failed, log error and return
        if correction_angle is None:
            self.error_logging(f"Could not determine valid rotation angle from response at index {index}: {response}")
            return

        # --- End Edit ---

        # If no rotation needed, just log and return
        if correction_angle == 0:
            self.error_logging(f"No rotation needed for index {index} based on response: {response}", level="INFO") # Log info level
            return

        try:
            # Open, rotate, and save the image file
            with Image.open(image_path_abs) as img:
                # # Ensure EXIF orientation is handled *before* applying AI rotation
                # img = ImageOps.exif_transpose(img)
                # Apply the correction rotation
                rotated_img = img.rotate(correction_angle, expand=True)
                 # Save back to the original path (overwrite)
                rotated_img.save(image_path_abs, "JPEG", quality=95) # Assume JPEG, adjust if needed

            # If the currently viewed page was rotated, reload its image in the canvas
            if index == self.page_counter:
                self.image_handler.load_image(image_path_abs)

        except Exception as e:
            self.error_logging(f"Error rotating image at index {index} ({image_path_abs}): {e}")

    def update_display_after_ai(self, index, updated_column_name):
        """
        Updates the DataFrame's Text_Toggle and refreshes the display 
        if the modified page is currently visible.
        Should be called by AIFunctionsHandler after updating a text column.
        """
        if self.main_df.empty or index < 0 or index >= len(self.main_df):
            self.error_logging(f"update_display_after_ai called with invalid index: {index}", level="WARNING")
            return

        self.error_logging(f"update_display_after_ai called for index {index}, column {updated_column_name}", level="INFO") # <-- Added INFO log

        try:
            # --- MODIFIED --- Force update Text_Toggle if new text exists
            new_text = self.main_df.loc[index, updated_column_name]
            if pd.notna(new_text) and str(new_text).strip():
                self.main_df.loc[index, 'Text_Toggle'] = updated_column_name
                self.error_logging(f"Forced Text_Toggle for index {index} to {updated_column_name}", level="INFO")
            # --- END MODIFIED ---

            # If the updated page is the one currently displayed, refresh the UI
            if index == self.page_counter:
                self.error_logging(f"Refreshing display for current page {index} via load_text() after AI update ({updated_column_name})", level="INFO")
                # --- MODIFIED --- Call load_text directly
                self.load_text()
                # --- END MODIFIED ---
            else: # <-- Added else block for logging
                self.error_logging(f"AI update for index {index} ({updated_column_name}) occurred, but current page is {self.page_counter}. No immediate refresh.", level="INFO") # <-- Added INFO log

        except Exception as e:
            self.error_logging(f"Error in update_display_after_ai for index {index}, column {updated_column_name}: {e}", level="ERROR")


# Callbacks

    def on_relevance_change(self, event=None):
        """Callback function when the relevance dropdown selection changes."""
        if self.main_df.empty or self.page_counter >= len(self.main_df):
            return

        index = self.page_counter
        selected_relevance = self.relevance_var.get()

        # Update the 'Relevance' column in the DataFrame for the current row
        if 'Relevance' in self.main_df.columns:
            self.main_df.loc[index, 'Relevance'] = selected_relevance

# Function Handlers

    def find_and_replace(self, event=None):
        self.find_replace.update_main_df(self.main_df)
        self.find_replace.find_and_replace(event)

    def update_api_handler(self):
        # Ensure settings are loaded before updating handler
        if not hasattr(self, 'settings'):
            self.settings = Settings() # Initialize if missing
        self.api_handler = APIHandler(
            self.settings.openai_api_key,
            self.settings.anthropic_api_key,
            self.settings.google_api_key,
            self
        )

    def edit_single_image(self):
        if self.main_df.empty:
            messagebox.showerror("Error", "No images have been loaded. Please load some images first.")
            return

        # Hide the main window
        self.withdraw()

        # Define temp directory for this single edit session
        # Use self.edit_temp_directory which should be initialized
        single_temp_dir = os.path.join(self.edit_temp_directory, f"edit_{self.page_counter}")
        # Clear and recreate directory for a clean session
        if os.path.exists(single_temp_dir):
            shutil.rmtree(single_temp_dir)
        os.makedirs(single_temp_dir, exist_ok=True)


        try:
            # Copy the current image to temp directory (resolve path first)
            current_image_path_rel = self.main_df.loc[self.page_counter, 'Image_Path']
            current_image_path_abs = self.get_full_path(current_image_path_rel)

            if not current_image_path_abs or not os.path.exists(current_image_path_abs):
                 raise FileNotFoundError(f"Current image not found: {current_image_path_abs}")

            # Use a simple name in the temp dir
            temp_image_name = "current_image" + os.path.splitext(current_image_path_abs)[1]
            temp_image_path = os.path.join(single_temp_dir, temp_image_name)

            shutil.copy2(current_image_path_abs, temp_image_path)

            # Create an instance of ImageSplitter with the specific temp directory
            image_splitter = ImageSplitter(single_temp_dir)

            # Wait for the ImageSplitter window to close
            self.wait_window(image_splitter)

            if image_splitter.status == "saved":
                # Pass the original RELATIVE path for context if needed
                # Call the method now located in DataOperations
                self.data_operations.process_edited_single_image(current_image_path_rel)
            elif image_splitter.status == "discarded":
                messagebox.showinfo("Cancelled", "Image editing discarded.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while editing the image: {str(e)}")
            self.error_logging(f"Error in edit_single_image: {str(e)}")


        finally:
            # Clean up the specific temp directory for this edit session
            if os.path.exists(single_temp_dir):
                try:
                    shutil.rmtree(single_temp_dir, ignore_errors=True)
                except Exception as e_clean:
                    self.error_logging(f"Error cleaning up single edit temp dir {single_temp_dir}: {e_clean}")

            # Show the main window again
            self.deiconify()
            self.lift() # Bring window to front

    def edit_all_images(self):
        if self.main_df.empty:
            messagebox.showerror("Error", "No images have been loaded. Please load some images first.")
            return

        if not messagebox.askyesno("Warning",
                                    "This action will replace all current images and potentially reset text data with the edited versions. "
                                    "All existing text associated with the original images might be lost or misaligned. This action cannot be undone. "
                                    "Do you want to continue?"):
            return

        self.withdraw()
        # Use the general edit_temp directory for this process
        all_temp_dir = os.path.join(self.edit_temp_directory, "edit_all")

        try:
            # Clean up existing temp directory if it exists
            if os.path.exists(all_temp_dir):
                shutil.rmtree(all_temp_dir)

            # Create fresh temp directory
            os.makedirs(all_temp_dir, exist_ok=True)

            # Copy all images to temp directory using absolute paths
            image_map = {} # Store mapping from temp name back to original index
            for index, row in self.main_df.iterrows():
                current_image_path_rel = row['Image_Path']
                # Use get_full_path to resolve relative paths
                current_image_path_abs = self.get_full_path(current_image_path_rel)

                if current_image_path_abs and os.path.exists(current_image_path_abs):
                    # Use index in temp name for sorting and mapping back
                    temp_image_name = f"{index:04d}{os.path.splitext(current_image_path_abs)[1]}"
                    temp_image_path = os.path.join(all_temp_dir, temp_image_name)
                    shutil.copy2(current_image_path_abs, temp_image_path)
                    image_map[temp_image_name] = index
                else:
                    self.error_logging(f"Image not found or path invalid for index {index}: {current_image_path_abs}", level="WARNING")
                    # Decide whether to raise error or just skip
                    # raise FileNotFoundError(f"Image not found: {current_image_path_abs}")

            if not image_map:
                 messagebox.showerror("Error", "No valid images found to edit.")
                 self.deiconify()
                 return


            image_splitter = ImageSplitter(all_temp_dir)
            self.wait_window(image_splitter)

            if image_splitter.status == "saved":
                 # Process the saved images from the "pass_images" directory
                 pass_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "util", "subs", "pass_images")

                 if not os.path.exists(pass_images_dir):
                     raise FileNotFoundError(f"pass_images directory not found at: {pass_images_dir}")

                 # Get the list of edited images
                 edited_images = sorted(
                    [f for f in os.listdir(pass_images_dir) if f.lower().endswith((".jpg", ".jpeg"))],
                     key=natural_sort_key # Sort naturally
                 )

                 if not edited_images:
                     messagebox.showwarning("No Edits", "No edited images were found in the output directory.")
                 else:
                    # --- Replace DataFrame content ---
                    # Store the current project directory before reset
                    project_dir = getattr(self, 'project_directory', None)
                    
                    self.reset_application() # Clear existing DF and state
                    
                    # Restore the project directory and images directory after reset
                    if project_dir:
                        self.project_directory = project_dir
                        self.images_directory = os.path.join(self.project_directory, "images")
                        os.makedirs(self.images_directory, exist_ok=True)

                    new_rows_list = []
                    for i, edited_image_file in enumerate(edited_images):
                        edited_image_path_abs = os.path.join(pass_images_dir, edited_image_file)

                        # Define new path in the project's images directory
                        new_image_name = f"{i+1:04d}_p{i+1:03d}.jpg" # New sequential naming
                        target_image_path_abs = os.path.join(self.images_directory, new_image_name)
                        target_image_path_rel = self.get_relative_path(target_image_path_abs)

                        # Copy edited image to project images directory
                        shutil.copy2(edited_image_path_abs, target_image_path_abs)

                        # Create new row for the DataFrame
                        new_row_data = {
                            "Index": i,
                            "Page": f"{i+1:04d}_p{i+1:03d}",
                            "Original_Text": "", "Corrected_Text": "", "Formatted_Text": "",
                            "Translation": "", "Separated_Text": "",
                            "Image_Path": target_image_path_rel, # Relative path
                            "Text_Path": "", # Text path is reset
                            "Text_Toggle": "None",
                            "People": "", "Places": "", "Errors": "", "Errors_Source": "", "Relevance": ""
                            # Add other columns initialized as empty
                        }
                        # Ensure all DF columns exist
                        for col in self.main_df.columns:
                            if col not in new_row_data:
                                 new_row_data[col] = ""
                        new_rows_list.append(new_row_data)

                    # Recreate the DataFrame
                    self.main_df = pd.DataFrame(new_rows_list)

                    # Clean up pass_images directory
                    for file in edited_images: # Use the list we already have
                        file_path = os.path.join(pass_images_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e_clean:
                            self.error_logging(f"Warning: Failed to clean up temp file {file_path}: {e_clean}")

                    # Refresh display to show the first new image
                    if not self.main_df.empty:
                         self.page_counter = 0
                         self.refresh_display()
                    else: # Should not happen if edited_images was not empty
                         self.counter_update()

                    messagebox.showinfo("Success", f"Successfully replaced project with {len(edited_images)} edited images.")


            elif image_splitter.status == "discarded":
                messagebox.showinfo("Cancelled", "Image editing was cancelled. No changes were made.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while editing the images: {str(e)}")
            self.error_logging(f"Error in edit_all_images: {str(e)}")
        finally:
            # Clean up the main temp directory for this process
            if os.path.exists(all_temp_dir):
                try:
                    shutil.rmtree(all_temp_dir, ignore_errors=True)
                except Exception as e_clean:
                    self.error_logging(f"Error cleaning up edit_all temp dir {all_temp_dir}: {e_clean}")

            self.deiconify()
            self.lift() # Bring window to front

    def apply_document_separation(self):
        """Apply document separation based on ***** markers and replace main_df with the compiled documents."""
        try:
            # Check if any pages have no recognized text (using find_right_text as a proxy)
            unrecognized_pages = []
            for index in self.main_df.index:
                 text = self.data_operations.find_right_text(index)
                 if not text.strip():
                      unrecognized_pages.append(index + 1) # Store 1-based page number

            if unrecognized_pages:
                page_list = ", ".join(map(str, unrecognized_pages[:5])) # Show first 5
                if len(unrecognized_pages) > 5: page_list += "..."
                warning_message = (
                    f"Warning: {len(unrecognized_pages)} page(s) (e.g., {page_list}) have no usable text "
                    f"in their selected display mode ('{self.text_display_var.get()}') and might be lost or misaligned during separation. "
                    "Continue anyway?"
                )
                if not messagebox.askyesno("Potentially Empty Pages", warning_message):
                    return

            from util.SeparateDocuments import apply_document_separation
            # Disable buttons during potentially long operation
            self.toggle_button_state()
            apply_document_separation(self) # Pass the App instance
        finally:
            # Make sure buttons are re-enabled regardless of success or failure
            # Check current state before toggling
            if self.button1['state'] == "disabled":
                self.toggle_button_state()

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        # Log the error
        error_message = f"Critical error in main application scope: {str(e)}"
        # Try to write to error log if possible
        try:
            # Ensure util directory exists
            util_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "util")
            os.makedirs(util_dir, exist_ok=True)
            error_logs_path = os.path.join(util_dir, "error_logs.txt")
            with open(error_logs_path, "a", encoding='utf-8') as f:
                f.write(f"{datetime.now()}: CRITICAL: {error_message}\n{traceback.format_exc()}\n")
        except Exception as log_e:
            print(f"Failed to write to error log: {log_e}")
        # Try to show error message box
        try:
            messagebox.showerror("Critical Error",
                f"The application encountered a critical error and needs to close:\n\n{error_message}\n\n"
                f"Please check the error_logs.txt file in the 'util' folder for details.")
        except Exception as msg_e:
             print(f"Failed to show error messagebox: {msg_e}")
        # Ensure app closes if possible
        try:
            if 'app' in locals() and isinstance(app, tk.Tk):
                app.quit()
                app.destroy()
        except Exception as destroy_e:
            print(f"Failed to destroy app window: {destroy_e}")
            print(f"Failed to destroy app window: {destroy_e}")