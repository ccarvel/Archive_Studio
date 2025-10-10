# Archive Studio with macOS Silicon and Intel Releases
A desktop application designed to assist historians, archivists, and researchers in processing, transcribing, and analyzing historical documents using Large Language Models (LLMs) via API services (OpenAI, Google, and Anthropic).

#### NOTE: This fork of [Mark Humphries'](https://github.com/mhumphries2323) [Archive Studio](https://github.com/mhumphries2323/ArchiveStudio) contains built binaries for both Intel (x86_64) and Silicon (aarch64) macOS (tested with Sequoia 15.7). 

#### It also contains support for TIF/TIFF and JP2 images via drag and drop to the GUI or from File > Import Images from Folder; TIF/TIFF, JP2, and PNG files imported via the File menu are converted to .jpg images and the originals are moved to a subfolder named 'original_non-jpeg_images'. 

---

## Table of Contents

*   [Overview](#overview)
*   [Features](#features)
*   [User Manual](#user-manual)
*   [Requirements](#requirements)
*   [Installation](#installation)
*   [API Key Setup](#api-key-setup)
*   [Basic Workflow](#basic-workflow)
*   [Configuration](#configuration)
*   [Exporting](#exporting)
*   [License](#license)
*   [Citation](#citation)
*   [Authors](#authors)
*   [Disclaimer](#disclaimer)
*   [Contributing](#contributing)
*   [Acknowledgments](#acknowledgments)

---

## Overview

Archive Studio integrates document viewing, text editing, and Artificial Intelligence (AI) capabilities via external APIs (OpenAI, Anthropic, Google) to facilitate tasks such as transcription (HTR), text correction, relevance analysis, metadata extraction, document separation ("chunking"), and content analysis for historical documents.

It provides a structured environment for managing document images and their associated textual data through various processing stages, supporting batch processing to significantly speed up research workflows. The application is designed primarily for academic and research purposes, helping users transform raw historical documents into structured, analyzable data.

## Features

*   **Multi-API AI Processing:** Leverage models from OpenAI, Google (Gemini), and Anthropic for various tasks.
*   **Handwritten Text Recognition (HTR):** Generate initial transcriptions from document images.
*   **AI-Powered Text Enhancement:** Correct, format, and translate existing transcriptions.
*   **Metadata & Content Analysis:** Extract names, places, potential errors, and assess document relevance based on custom criteria.
*   **Names & Places Collation Tool:** Standardize extracted entities across the project.
*   **Document Separation Workflow:** Automatically identify logical document breaks (e.g., letters, diary entries) and reorganize the project accordingly.
*   **Integrated Interface:** Resizable side-by-side image viewer (zoom/pan) and text editor.
*   **Multiple Text Version Management:** Work with and compare different stages of text (Original, Corrected, Formatted, Translation, etc.).
*   **Highlighting:** Visualize names, places, errors, and differences between text versions.
*   **Comprehensive Import Options:** Import PDFs (extracting images and embedded text), folders of images, or paired image/text files (e.g., from Transkribus). Drag-and-drop support for PDFs and images.
*   **Image Preprocessing Tool:** Built-in utility to split (multi-page spreads), crop, straighten (manual/auto), and rotate images before AI processing. Includes batch capabilities.
*   **Configurable AI Presets:** Customize prompts, models, and parameters for all AI functions.
*   **Project Management:** Create, save, and load projects with associated images and text data.
*   **Batch Processing:** Apply AI functions to multiple pages concurrently with progress tracking.
*   **Multiple Export Formats:** Export results as single/separate TXT files, searchable PDFs (image + hidden text), or highly configurable CSV files for metadata and analysis.

## User Manual

A comprehensive **ArchiveStudio User Manual (v1.0 beta)** is available in the repository:

**[ArchiveStudio_User_Manual_v1.0_beta.pdf](Manual.pdf)** 

This manual provides detailed instructions on:

*   Installation and Setup
*   **Obtaining and configuring API Keys (Section 3.0)** - **Crucial for AI features!**
*   Understanding API costs and tokenization
*   Navigating the interface
*   Using all features, including import/export, AI processing, presets, and the Image Preprocessing Tool.
*   Keyboard shortcuts

## Requirements

*   **Operating System:** Microsoft Windows 10 or later.
*   **Internet Connection:** Required for using AI features via APIs.
*   **API Keys:** Active API keys from at least one supported provider:
    *   OpenAI ([platform.openai.com](https://platform.openai.com/))
    *   Anthropic ([console.anthropic.com](https://console.anthropic.com/))
    *   Google AI Studio ([aistudio.google.com](https://aistudio.google.com/))
    *   **Note:** API usage incurs costs billed directly by the providers. See Section 3.2 of the manual.

## Installation

1.  Download the latest `ArchiveStudio.exe` executable file and the accompanying `util` directory from the [Releases page](https://github.com/mhumphries2323/ArchiveStudio/releases) or the repository. <!-- **IMPORTANT:** Update this link -->
2.  Create a dedicated directory for the application (e.g., `C:\ArchiveStudio`).
3.  Move the `ArchiveStudio.exe` file AND the entire `util` directory (with all its contents) into the directory you created. The program requires the files within the `util` folder to function correctly.
4.  **Antivirus Software:** As unsigned software, ArchiveStudio might be flagged by Windows Defender or other antivirus programs. You may need to create an exception to allow the program to run. Ensure the entire program directory is allowed.

## API Key Setup

Using the AI features of ArchiveStudio **requires API keys** from OpenAI, Anthropic, and/or Google.

1.  **Obtain Keys:** Follow the instructions in **Section 3.3 of the User Manual** to create accounts and generate API keys on the respective provider platforms.
2.  **Configure Billing:** Set up billing information or purchase credits on each platform you intend to use.
3.  **Set Usage Limits (Recommended):** Configure spending limits within your provider accounts to prevent unexpected costs.
4.  **Enter Keys in ArchiveStudio:** Launch ArchiveStudio, go to `File -> Settings -> API Settings`, and paste your keys into the corresponding fields. Keys are stored locally on your machine.

**Warning:** API usage is billed by the AI providers based on data processed (tokens for text, per image/tile for images). Familiarize yourself with provider pricing before processing documents. See **Section 3.2 of the User Manual** for more details on costs and tokenization.

## Basic Workflow

1.  **Launch** the `ArchiveStudio.exe` application.
2.  **Create a New Project** (`File -> New Project`) or **Open an Existing Project** (`File -> Open Project`).
3.  **Import Documents:**
    *   `File -> Import PDF...` (extracts images and embedded text)
    *   `File -> Import Images from Folder...`
    *   `File -> Import Text and Images...` (for pre-existing transcriptions)
    *   Or, **Drag and Drop** PDF/image files onto the window.
4.  **(Optional) Preprocess Images:** Use the `Tools -> Edit Current/All Images` (Ctrl+I / Ctrl+Shift+I) tool to split, crop, or straighten images.
5.  **Process with AI:** Use functions under the `Process` menu (e.g., `Recognize Text (HTR)`, `Correct Text`, `Get Names and Places`, `Identify Document Separators`). Configure presets in `File -> Settings`.
6.  **Review and Edit:** Navigate pages, view different text versions, edit text directly, use highlighting features (`Highlights` menu), and standardize entities (`Tools -> Edit Names & Places`).
7.  **(Optional) Apply Document Separation:** If separators were identified, use `Process -> Apply Document Separation` to restructure the project.
8.  **Export Results:** Use `File -> Export Text...` or `File -> Export CSV...` to save your work in desired formats.
9.  **Save Project:** Use `File -> Save Project` (Ctrl+S) or `File -> Save Project As...`.

## Configuration

Application behavior, especially AI functions, is controlled via the **Settings** window (`File -> Settings`):

*   **API Settings:** Enter API keys and set batch processing size.
*   **Model Settings:** Define available AI models and configure import options (e.g., auto-orientation check).
*   **Preset Management:** Crucial for customizing AI tasks. Create, edit, and manage presets for:
    *   `Metadata`: Configure metadata extraction for CSV export.
    *   `Sequential Metadata`: Configure sequential date/place analysis.
    *   `Analysis`: Configure relevance analysis and potentially other tasks.
    *   `Document Separation`: Define how AI identifies document breaks.
    *   `Function`: Customize core AI tasks (HTR, Correction, Errors, Names/Places, Rotation, Translation).
    *   `Format`: Define text reformatting rules.
*   Settings can be **Saved**, **Loaded**, **Exported** (`.psf`), **Imported**, or **Restored to Defaults**.

Settings are stored locally in `settings.json` (typically in `C:\Users\[YourUsername]\AppData\Roaming\ArchiveStudio`).

## Exporting

ArchiveStudio offers various export options (`File` menu):

*   **Single Text File:** Combines text from all documents/pages.
*   **Separate Text Files:** One `.txt` file per document/page.
*   **PDF with Searchable Text:** Creates a PDF with document images overlaid with invisible, searchable text.
*   **CSV (Metadata Export):** Highly configurable export for metadata and text, suitable for databases or spreadsheets. Options include:
    *   Filtering by relevance.
    *   Selecting the source text version.
    *   AI-powered metadata generation based on presets.
    *   Sequential date/place analysis.
    *   Custom citation and author information.

See **Section 13.0 of the User Manual** for detailed export options.

## License

License: **CC BY-NC 4.0**

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

This means you are free to:

*   **Share** — copy and redistribute the material in any medium or format
*   **Adapt** — remix, transform, and build upon the material

Under the following terms:

*   **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
*   **NonCommercial** — You may not use the material for commercial purposes.

## Citation

If you use this software in your research, please cite:

> Mark Humphries and Lianne C. Leddy, 2025. ArchiveStudio 1.0 Beta. Department of History: Wilfrid Laurier University.

If you wish to cite the paper that explores related research:

> Mark Humphries, Lianne C. Leddy, Quinn Downton, Meredith Legace, John McConnell, Isabella Murray, and Elizabeth Spence. Unlocking the Archives: Using Large Language Models to Transcribe Handwritten Historical Documents. *Preprint: [Add Link or DOI when available]*.

## Authors

*   **Mark Humphries** (Programming, Funding, and Historical Work) - Wilfrid Laurier University, Waterloo, Ontario
*   **Lianne Leddy** (Funding, Historical Work, and Testing) - Wilfrid Laurier University, Waterloo, Ontario

*(User Manual written by Mark Humphries and Gemini-1.0-Pro)*

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

The authors assume no liability for its use or any damages resulting from its use, including costs incurred from API usage. Users are solely responsible for managing their API keys and associated costs.

## Contributing

This project is primarily developed for academic and research purposes within its specific context. Direct contributions via pull requests are generally not solicited. However, interested potential collaborators for academic research purposes are welcome to contact the primary author.

## Acknowledgments

This software utilizes APIs provided by:

*   [OpenAI](https://openai.com/)
*   [Google (Gemini Models)](https://ai.google.dev/)
*   [Anthropic (Claude Models)](https://www.anthropic.com/)

Their powerful language models make the core functionality of ArchiveStudio possible.
