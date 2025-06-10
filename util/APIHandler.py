# util/APIHandler.py

# This file contains the APIHandler class, which is used to handle
# the API calls for the application.

import asyncio
import base64
import os
from pathlib import Path
from PIL import Image

# OpenAI API
from openai import OpenAI
import openai

# Anthropic API
from anthropic import AsyncAnthropic
import anthropic

# Google API
import google.generativeai as genai
from google.generativeai import types # For Part, Content, GenerationConfig
# It's good practice to also import specific exception types if you want to catch them
# from google.api_core import exceptions as google_exceptions

# Import ErrorLogger
from util.ErrorLogger import log_error

class APIHandler:
    def __init__(self, openai_api_key, anthropic_api_key, google_api_key, app=None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.google_api_key = google_api_key
        self.app = app  # Reference to main app for error logging
        
        self.is_google_configured = False
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.is_google_configured = True
                print("[INFO] Google Generative AI SDK configured successfully.")
            except Exception as e:
                # Use self.log_error if app context is available, otherwise print
                error_message = f"Failed to configure Google Generative AI SDK: {e}"
                if self.app and hasattr(self.app, 'base_dir') and hasattr(self.app, 'log_level'):
                    self.log_error("Google SDK Configuration Error", error_message)
                else:
                    print(f"[ERROR] {error_message}")
        else:
            print("[INFO] Google API key not provided. Google Generative AI features will be unavailable.")

    def log_error(self, error_message, additional_info=None):
        """Log errors using ErrorLogger if app is available, otherwise silently continue"""
        if self.app and hasattr(self.app, 'base_dir') and hasattr(self.app, 'log_level'):
            log_error(self.app.base_dir, self.app.log_level, error_message, additional_info, level="ERROR")
        
    async def route_api_call(self, engine, system_prompt, user_prompt, temp, 
                           image_data=None, text_to_process=None, val_text=None, 
                           index=None, is_base64=True, formatting_function=False, 
                           api_timeout=80, job_type=None, job_params=None):
        """
        Routes the API call to the appropriate service based on the engine name.
        
        Args:
            engine: Model to use (gpt/claude/gemini)
            system_prompt: System instructions for the AI
            user_prompt: User instructions for the AI
            temp: Temperature setting for model output
            image_data: Optional image data (single image or list of tuples of (path, label) for Gemini)
            text_to_process: Text to be processed and inserted into prompt
            val_text: Validation text to check in response
            index: Document index for tracking
            is_base64: Whether images are base64 encoded (ignored for Gemini)
            formatting_function: Whether to use user_prompt directly or format it
            api_timeout: Timeout in seconds for API call
            job_type: Type of job (e.g., "Metadata")
            job_params: Additional parameters for the job
        """
        required_headers = job_params.get("required_headers") if job_type == "Metadata" and job_params else None
        
        if image_data:
            if isinstance(image_data, list):
                try:
                    print(f"[DEBUG] APIHandler.route_api_call received {len(image_data)} images for engine {engine}.")
                    # Example of how to print labels if they exist:
                    # print(f"Labels: {[item[1] if isinstance(item, tuple) and len(item) > 1 else 'No Label' for item in image_data]}")
                except Exception:
                    print(f"[DEBUG] APIHandler.route_api_call received image_data (list) but could not extract details for engine {engine}.")
            else:
                print(f"[DEBUG] APIHandler.route_api_call received image_data of type {type(image_data)} for engine {engine}")
        else:
            print(f"[DEBUG] APIHandler.route_api_call received no image_data for engine {engine}.")
        
        if "gpt" in engine.lower() or "o1" in engine.lower() or "o3" in engine.lower():
            return await self.handle_gpt_call(system_prompt, user_prompt, temp, 
                                           image_data, text_to_process, val_text, 
                                           engine, index, is_base64, formatting_function, 
                                           api_timeout, job_type, required_headers)
        elif "gemini" in engine.lower():
            return await self.handle_gemini_call(system_prompt, user_prompt, temp, 
                                              image_data, text_to_process, val_text, 
                                              engine, index, is_base64, formatting_function, # is_base64 is effectively ignored by Gemini handler
                                              api_timeout, job_type, required_headers)
        elif "claude" in engine.lower():
            return await self.handle_claude_call(system_prompt, user_prompt, temp, 
                                              image_data, text_to_process, val_text, 
                                              engine, index, is_base64, formatting_function, 
                                              api_timeout, job_type, required_headers)
        else:
            self.log_error(f"Unsupported engine: {engine}")
            raise ValueError(f"Unsupported engine: {engine}")
    
    def _prepare_gpt_messages(self, system_prompt, user_prompt, image_data):
        """Prepare messages for GPT API with system prompt and image content if present"""
        is_o_series_model = "o1" in system_prompt.lower() or "o3" in system_prompt.lower() # engine might be better here
        # Corrected to check engine name for o-series model type.
        # This check might be more robust if it used self.engine or passed engine to this method.
        # Assuming system_prompt containing o1/o3 is a convention.
        
        role_key = "developer" if is_o_series_model else "system"
        
        if not image_data:
            return [
                {"role": role_key, "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
        
        if isinstance(image_data, str): # Assuming single base64 string
            return [
                {"role": role_key, "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ]
        
        # Multiple images case (list of (base64_img_string, label))
        content = [{"type": "text", "text": user_prompt}]
        for img_base64, label in image_data: # Ensure image_data is in this format for GPT
            if label:
                content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "high"
                }
            })
        
        return [
            {"role": role_key, "content": system_prompt},
            {"role": "user", "content": content}
        ]
    
    async def handle_gpt_call(self, system_prompt, user_prompt, temp, image_data, 
                            text_to_process, val_text, engine, index, 
                            is_base64=True, formatting_function=False, api_timeout=25.0,
                            job_type=None, required_headers=None):
        # (Keep existing GPT call logic - ensure image_data is base64 encoded string or list of (base64, label))
        # This method appears largely correct based on its existing structure.
        client = OpenAI(api_key=self.openai_api_key, timeout=api_timeout)
        
        populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)
        max_tokens = 2000 if job_type == "Metadata" else (200 if "pagination" in user_prompt.lower() else 1500)
        max_retries = 5 if job_type == "Metadata" else 3
        retries = 0
        is_o_series_model = "o1" in engine.lower() or "o3" in engine.lower() # Check engine name
        
        while retries < max_retries:
            try:
                # Ensure image_data is correctly formatted for _prepare_gpt_messages
                # If image_data comes from prepare_image_data, it should be base64 for GPT.
                messages = self._prepare_gpt_messages(system_prompt, populated_user_prompt, image_data)
                
                api_params = {
                    "model": engine,
                    "messages": messages,
                }
                
                if is_o_series_model:
                    api_params["response_format"] = {"type": "text"}
                    # "reasoning_effort" might be specific or deprecated, check OpenAI docs if issues arise
                    api_params["reasoning_effort"] = "low" 
                else:
                    api_params["temperature"] = temp
                    api_params["max_tokens"] = max_tokens
                
                completion = client.chat.completions.create(**api_params)
                response = completion.choices[0].message.content
                validation_result = self._validate_response(response, val_text, index, job_type, required_headers)
                
                if validation_result[0] == "Error" and retries < max_retries - 1:
                    if job_type == "Metadata" and not is_o_series_model:
                        api_params["temperature"] = min(0.9, float(temp) + (retries * 0.1))
                        if retries >= 2:
                            api_params["max_tokens"] = min(4000, max_tokens + 500)
                    
                    retries += 1
                    await asyncio.sleep(1 * (1.5 ** retries))
                    continue
                
                return validation_result

            except (openai.APITimeoutError, openai.APIError) as e:
                self.log_error(f"GPT API Error with {engine} for index {index}", f"{str(e)}")
                retries += 1
                if retries == max_retries:
                    return "Error", index
                await asyncio.sleep(1 * (1.5 ** retries))
            except Exception as e: # Catch any other unexpected error
                self.log_error(f"Unexpected error in GPT call with {engine} for index {index}", f"{str(e)}")
                return "Error", index # Stop retrying on unexpected errors for now
        return "Error", index # Fallback if loop finishes without returning


    async def handle_gemini_call(self, system_prompt, user_prompt, temp, image_data, 
                                text_to_process, val_text, engine, index, 
                                is_base64=True, formatting_function=False, api_timeout=120.0,
                                job_type=None, required_headers=None):
        """Handle API calls to Google Gemini models"""
        if not self.is_google_configured:
            self.log_error("Google API key not configured or configuration failed. Skipping Gemini call.", 
                           f"Engine: {engine}, Index: {index}")
            return "Error", index

        model = genai.GenerativeModel(
            model_name=engine,
            system_instruction=system_prompt # system_prompt should be a string here
        )
        
        try:
            temperature_float = float(temp)
        except ValueError:
            self.log_error(f"Invalid temperature value for Gemini: {temp}. Defaulting to 0.5.", f"Index: {index}")
            temperature_float = 0.5

        generation_config = types.GenerationConfig( # This 'types' is correct for GenerationConfig
            temperature=temperature_float,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        
        populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)
        
        max_retries = 5 if job_type == "Metadata" else 3
        retries = 0
        
        while retries < max_retries:
            try:
                parts = [] # This will be a list of strings and/or File objects
                
                # Handle image data
                if image_data:
                    if isinstance(image_data, (str, Path)): # Single image path
                        image_path_str = str(image_data)
                        print(f"[DEBUG] Gemini: Uploading single image: {image_path_str}")
                        try:
                            uploaded_file = genai.upload_file(path=image_path_str)
                            parts.append(uploaded_file) # CORRECTED: Append File object directly
                        except Exception as e:
                            self.log_error(f"Gemini: Failed to upload single image {image_path_str} for index {index}", str(e))
                            # Depending on desired behavior, could return "Error" or continue without image
                            pass # Or raise, or return
                            
                    elif isinstance(image_data, list): # List of (img_path, label)
                        for item_index, item in enumerate(image_data):
                            if not (isinstance(item, tuple) and len(item) == 2):
                                self.log_error(f"Gemini: Invalid item format in image_data list at pos {item_index}. Expected (path, label). Got: {item}", f"Index: {index}")
                                continue

                            img_path, label = item
                            if not isinstance(img_path, (str, Path)):
                                self.log_error(f"Gemini: Image path is not a string or Path object: {img_path} at pos {item_index}", f"Index: {index}")
                                continue

                            img_path_str = str(img_path)
                            if label and isinstance(label, str): # Add text label part first
                                parts.append(label) # CORRECTED: Append string directly
                            
                            print(f"[DEBUG] Gemini: Uploading image from list: {img_path_str} (Label: {label})")
                            try:
                                uploaded_file = genai.upload_file(path=img_path_str)
                                parts.append(uploaded_file) # CORRECTED: Append File object directly
                            except Exception as e:
                                self.log_error(f"Gemini: Failed to upload image {img_path_str} from list for index {index}", str(e))
                                # Continue to next image or part
                                pass
                    else:
                        self.log_error(f"Gemini: Unsupported image_data type: {type(image_data)}", f"Index: {index}")
                
                # Add the main user prompt text
                if populated_user_prompt and populated_user_prompt.strip(): # Ensure not just whitespace
                    parts.append(populated_user_prompt.strip()) # CORRECTED: Append string directly

                if not parts:
                    self.log_error("Gemini: No content parts to send (no prompt and no valid images).", f"Index: {index}")
                    return "Error", index

                print(f"\n[GEMINI API CALL - Index {index}]")
                print(f"Model: {engine}")
                # For debugging the parts list:
                # print(f"Request Parts (types): {[type(p) for p in parts]}")
                # print(f"Request Parts (content preview): {[str(p)[:100] if isinstance(p, str) else p.uri if hasattr(p, 'uri') else type(p) for p in parts]}")

                response_text = ""
                response_stream = model.generate_content(
                    contents=parts, # This is now a list of strings and/or File objects
                    generation_config=generation_config,
                    stream=True,
                    request_options={'timeout': float(api_timeout)}
                )
                
                for chunk in response_stream:
                    # Check if chunk has 'text' attribute and it's not None
                    if hasattr(chunk, 'text') and chunk.text is not None:
                        response_text += chunk.text
                    # Sometimes, especially with errors or empty responses, chunk.parts might be empty
                    # or parts might not have text. Adding a more robust check:
                    elif hasattr(chunk, 'parts'):
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text is not None:
                                response_text += part.text
                
                print(f"[Gemini API Raw Response - Index {index}]:\n{response_text[:500].strip()}...\n")

                validation_result = self._validate_response(response_text, val_text, index, job_type, required_headers)
                
                if validation_result[0] == "Error" and retries < max_retries - 1:
                    if job_type == "Metadata":
                        current_temp = generation_config.temperature
                        generation_config.temperature = min(0.9, (current_temp if current_temp is not None else 0.5) + ((retries + 1) * 0.1))
                        print(f"[DEBUG] Gemini Retry {retries+1}/{max_retries}: Adjusted temperature to {generation_config.temperature} for index {index}")
                    
                    retries += 1
                    await asyncio.sleep(1 * (1.5 ** retries))
                    continue
                
                return validation_result

            except Exception as e:
                print(f"[Gemini API Exception - Index {index}]: {type(e).__name__} - {e}")
                self.log_error(f"Gemini API Error with {engine} for index {index}", f"{type(e).__name__}: {str(e)}")
                # Log traceback for unexpected errors if possible/needed for debugging
                # import traceback
                # self.log_error(f"Gemini API Traceback", traceback.format_exc())
                retries += 1
                if retries == max_retries:
                    return "Error", index
                await asyncio.sleep(1 * (1.5 ** retries))
        return "Error", index

    async def handle_claude_call(self, system_prompt, user_prompt, temp, image_data, 
                                text_to_process, val_text, engine, index, 
                                is_base64=True, formatting_function=False, api_timeout=120.0,
                                job_type=None, required_headers=None):
        # (Keep existing Claude call logic - ensure image_data is base64 or list of (base64, label))
        # This method appears largely correct based on its existing structure.
        async with AsyncAnthropic(api_key=self.anthropic_api_key, 
                                max_retries=0, # Retries handled manually below
                                timeout=api_timeout) as client: # Anthropic client takes timeout directly
            
            populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)

            if job_type == "Metadata":
                max_tokens = 2000
            elif "Pagination:" in user_prompt.lower() or "Split Before:" in user_prompt: # Case-sensitive check
                max_tokens = 200
            elif "extract information" in user_prompt.lower():
                max_tokens = 1500
            else:
                max_tokens = 1200

            # Prepare message content with images if present
            # For Claude, image_data is expected as base64 string or list of (base64_data, label)
            content = []
            
            if image_data:
                if isinstance(image_data, str): # Single base64 image string
                    content.append({"type": "text", "text": "Document Image:"}) # Optional: context for single image
                    content.append({
                        "type": "image",
                        "source": { "type": "base64", "media_type": "image/jpeg", "data": image_data }
                    })
                elif isinstance(image_data, list): # List of (base64_img, label)
                    for img_base64, label in image_data:
                        if label and isinstance(label, str):
                            content.append({"type": "text", "text": label})
                        
                        # Ensure img_base64 is a string. If it's bytes, encode it.
                        # (prepare_image_data should handle this for Claude)
                        if isinstance(img_base64, bytes):
                            img_base64_str = base64.b64encode(img_base64).decode('utf-8')
                        elif isinstance(img_base64, str):
                            img_base64_str = img_base64
                        else:
                            self.log_error(f"Claude: Invalid image data type in list for index {index}", f"Type: {type(img_base64)}")
                            continue # Skip this image

                        content.append({
                            "type": "image",
                            "source": { "type": "base64", "media_type": "image/jpeg", "data": img_base64_str }
                        })
            
            # Add the user prompt text at the end
            if populated_user_prompt and populated_user_prompt.strip():
                content.append({"type": "text", "text": populated_user_prompt.strip()})

            if not content: # Should not happen if there's always a user_prompt
                self.log_error("Claude: No content to send (no prompt and no valid images).", f"Index: {index}")
                return "Error", index

            max_retries = 5 if job_type == "Metadata" else 3
            retries = 0
            current_temp = float(temp) # Ensure temp is float
            current_max_tokens = max_tokens
            
            while retries < max_retries:
                try:
                    message = await client.messages.create(
                        max_tokens=current_max_tokens,
                        messages=[{"role": "user", "content": content}],
                        system=system_prompt,
                        model=engine,
                        temperature=current_temp,
                        # timeout on client.messages.create is also possible if client itself doesn't take it for all requests
                    )
                    
                    response = message.content[0].text # Assuming first content block is text
                    validation_result = self._validate_response(response, val_text, index, job_type, required_headers)
                    
                    if validation_result[0] == "Error" and retries < max_retries - 1:
                        if job_type == "Metadata":
                            current_temp = min(0.9, current_temp + ((retries + 1) * 0.1))
                            if retries >= 1: # Original logic was retries >= 2, adjust if needed
                                current_max_tokens = min(4000, current_max_tokens + 500)
                            print(f"[DEBUG] Claude Retry {retries+1}/{max_retries}: Temp {current_temp}, Tokens {current_max_tokens} for index {index}")
                        
                        retries += 1
                        await asyncio.sleep(1 * (1.5 ** retries))
                        continue
                    
                    return validation_result

                except (anthropic.APITimeoutError, anthropic.APIError) as e:
                    self.log_error(f"Claude API Error with {engine} for index {index}", f"{type(e).__name__}: {str(e)}")
                    retries += 1
                    if retries == max_retries:
                        return "Error", index
                    await asyncio.sleep(1 * (1.5 ** retries))
                except Exception as e: # Catch any other unexpected error
                    self.log_error(f"Unexpected error in Claude call with {engine} for index {index}", f"{type(e).__name__}: {str(e)}")
                    return "Error", index # Stop retrying on unexpected errors
            return "Error", index # Fallback if loop finishes

    def _validate_response(self, response, val_text, index, job_type=None, required_headers=None):
        # (Keep existing validation logic)
        # This method appears largely correct based on its existing structure.
        if not response: # Handles empty or None response
            self.log_error(f"Empty API response for index {index}", f"job_type: {job_type}, val_text: {val_text}")
            return "Error", index
            
        if not val_text or val_text == "None": # No specific validation string needed
            return response, index
            
        try:
            # Ensure response is a string for 'in' and 'split' operations
            if not isinstance(response, str):
                self.log_error(f"API response is not a string for index {index}", 
                               f"Type: {type(response)}, Response snippet: {str(response)[:100]}")
                return "Error", index

            if val_text in response:
                processed_response = response.split(val_text, 1)[1].strip()
                
                if job_type == "Metadata" and required_headers:
                    missing_headers = []
                    all_headers_present_but_empty = True # Assume true until content is found
                    
                    for header_name in required_headers:
                        # Case-insensitive check for header existence might be more robust
                        # For now, using exact match as in original logic.
                        header_pattern = f"{header_name}:" 
                        if header_pattern not in processed_response:
                            missing_headers.append(header_name)
                            continue # This header is missing
                            
                        # Check if header has actual content
                        try:
                            # Extract content for this specific header
                            # This logic can be complex if headers are multi-line or order varies.
                            # A simple split assumes header content is on the same line or immediately after.
                            # More robust parsing might use regex.
                            
                            # Find start of current header's content
                            start_idx = processed_response.find(header_pattern) + len(header_pattern)
                            
                            # Find start of the *next* known header to delimit current header's content
                            end_idx = len(processed_response) # Default to end of response
                            temp_processed_after_header = processed_response[start_idx:]

                            for next_hdr_name in required_headers:
                                if next_hdr_name == header_name: continue # Don't check against self
                                next_hdr_pattern = f"\n{next_hdr_name}:" # Assume headers are newline separated
                                pos = temp_processed_after_header.find(next_hdr_pattern)
                                if pos != -1 and (start_idx + pos) < end_idx:
                                     # This seems complex, let's simplify the check:
                                     # Check if the part after "Header:" and before the next known header (or EOL) is non-empty.
                                     pass # Original logic was a bit convoluted here.

                            # Simplified check: split by header, take next part, split by newline.
                            # This assumes single-line content for headers.
                            header_content_part = processed_response.split(header_pattern, 1)[1]
                            actual_content = header_content_part.split('\n', 1)[0].strip()

                            if actual_content and not actual_content.isspace():
                                all_headers_present_but_empty = False # Found at least one header with content
                                
                        except Exception as e:
                            self.log_error(f"Error checking content for header '{header_name}' at index {index}", f"{str(e)}")
                    
                    if missing_headers:
                        self.log_error(f"Missing required headers in metadata response for index {index}", 
                                       f"Missing: {missing_headers}, Required: {required_headers}, Response: {response[:300]}...")
                        return "Error", index
                    
                    if all_headers_present_but_empty and required_headers: # Only an error if headers were required
                        self.log_error(f"All required metadata headers are present but empty for index {index}", 
                                       f"Response: {response[:300]}...")
                        return "Error", index
                
                return processed_response, index
            else:
                self.log_error(f"Validation text '{val_text}' not found in response for index {index}", 
                               f"Response snippet: {response[:300]}...")
        except TypeError as e: # Should be less frequent with upfront type check
            self.log_error(f"Validation error (TypeError) for index {index}", 
                           f"Error: {e}, Response type: {type(response)}, Val_text type: {type(val_text)}")
            return "Error", index
        except Exception as e: # Catch any other validation error
            self.log_error(f"Unexpected validation error for index {index}",
                           f"Error: {type(e).__name__} - {e}, Val_text: {val_text}, Response: {response[:300]}...")
            return "Error", index

        return "Error", index


    def prepare_image_data(self, image_data, engine, is_base64=True):
        """
        Prepare image data in the format required by the specified engine.
        For Gemini: expects image_data to be a file path (str/Path) or a list of (file_path, label). Returns as is.
        For GPT/Claude: expects base64 encoded strings. Converts paths to base64.
        
        Args:
            image_data: Image path(s), data, or list of (path/data, label)
            engine: The AI model engine being used
            is_base64: Hint that output should be base64 (primarily for non-Gemini)
            
        Returns:
            Processed image data ready for the API, or None if input is None or processing fails.
        """
        if not image_data:
            return None

        # For Gemini, return the file paths directly (or list of (path, label))
        # The caller should ensure image_data is in the correct path format for Gemini.
        if "gemini" in engine.lower():
            # Basic validation for Gemini image_data structure
            if isinstance(image_data, (str, Path)):
                return image_data # Single path, fine
            elif isinstance(image_data, list):
                for item in image_data:
                    if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (str, Path))):
                        self.log_error(f"Gemini prepare_image_data: Invalid item in list: {item}. Expected (path, label).",
                                       f"Full image_data: {image_data}")
                        # Return None or raise error, as this indicates a problem upstream
                        return None # Or filter out bad items, or raise
                return image_data # List of (path, label), fine
            else:
                self.log_error(f"Gemini prepare_image_data: Unexpected image_data type: {type(image_data)}",
                               f"Expected str, Path, or list of (path, label). Data: {image_data}")
                return None # Data is not in expected format for Gemini


        # For GPT and Claude, images need to be base64 encoded.
        # The 'is_base64' flag here is more of an indicator of the target format.
        # The function's job is to ensure it *becomes* base64 if it's a path.
        
        # Handle single image case (path string)
        if isinstance(image_data, (str, Path)):
            # If it's already base64 (heuristically), pass through. Otherwise, encode path.
            # This heuristic is risky. Better to rely on input type or a flag.
            # For now, assume if it's a string, it *could* be a path OR already base64.
            # If it's a Path object, it's definitely a path.
            if isinstance(image_data, Path) or (isinstance(image_data, str) and os.path.exists(image_data)):
                 encoded = self.encode_image(str(image_data))
                 if not encoded:
                     self.log_error(f"prepare_image_data: Failed to encode single image path: {image_data}", engine)
                 return encoded
            elif isinstance(image_data, str): # Assumed to be already base64 if not a valid path
                return image_data 
            else: # Should not happen if previous checks are exhaustive
                self.log_error(f"prepare_image_data: Unhandled single image data type: {type(image_data)} for engine {engine}", image_data)
                return None


        # Handle multiple images case (list of (img_path_or_base64, label))
        if isinstance(image_data, list):
            processed_data = []
            for item_idx, item in enumerate(image_data):
                if not (isinstance(item, tuple) and len(item) == 2):
                    self.log_error(f"prepare_image_data: Invalid item in list at pos {item_idx} for {engine}. Expected (img_content, label). Got: {item}", image_data)
                    continue

                img_content, label = item
                
                if isinstance(img_content, (str, Path)):
                    # If it's a Path or an existing file string, encode it.
                    if isinstance(img_content, Path) or (isinstance(img_content, str) and os.path.exists(img_content)):
                        encoded_image = self.encode_image(str(img_content))
                        if encoded_image:
                            processed_data.append((encoded_image, label))
                        else:
                            self.log_error(f"prepare_image_data: Failed to encode image path in list: {img_content} for {engine}", f"Label: {label}")
                    elif isinstance(img_content, str): # Assumed to be already base64
                        processed_data.append((img_content, label))
                    else:
                        self.log_error(f"prepare_image_data: Unhandled image content type in list: {type(img_content)} for {engine}", f"Content: {img_content}")
                elif isinstance(img_content, bytes): # If raw bytes are passed, base64 encode them
                    try:
                        encoded_image = base64.b64encode(img_content).decode('utf-8')
                        processed_data.append((encoded_image, label))
                    except Exception as e:
                        self.log_error(f"prepare_image_data: Failed to base64 encode bytes for {engine}", f"Error: {e}, Label: {label}")
                else:
                    self.log_error(f"prepare_image_data: Invalid type for image content in list for {engine}: {type(img_content)}", f"Label: {label}")

            return processed_data
        
        self.log_error(f"prepare_image_data: Unhandled image_data structure for engine {engine}", f"Type: {type(image_data)}, Data: {image_data}")
        return None


    def encode_image(self, image_path_str):
        """Convert image file to base64 string"""
        try:
            # Optional: Add image validation/resizing here if needed, e.g., with PIL
            # Example:
            # img = Image.open(image_path_str)
            # img.verify() # Verify it's a valid image
            # Check size, format, etc.
            with open(image_path_str, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            self.log_error(f"Error encoding image: File not found.", f"Path: {image_path_str}")
            return None
        except IOError as e: # PIL.UnidentifiedImageError inherits from IOError
             self.log_error(f"Error encoding image: IO error or invalid image file.", f"Path: {image_path_str}, Error: {str(e)}")
             return None
        except Exception as e:
            self.log_error(f"Error encoding image: Unexpected error.", f"Path: {image_path_str}, Error: {str(e)}")
            return None
