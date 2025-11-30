# shared_utils.py

import requests
import logging
import replicate
import openai
import httpx
import time
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import base64
from together import Together
from openai import OpenAI
import re
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Please install it with 'pip install beautifulsoup4'")

# Load environment variables
load_dotenv()

# Initialize Anthropic client with API key
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize OpenAI client lazily to avoid proxy kwargs issues on import
openai_client = None
def get_openai_client():
    global openai_client
    if openai_client is None:
        http_client = httpx.Client()
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), http_client=http_client)
    return openai_client

# -------------------- Cloudflare Workers AI usage guard --------------------
_cf_usage_lock = threading.Lock()
_cf_usage_file = Path("logs") / "cf_ai_usage.json"

def _load_cf_usage() -> dict:
    try:
        if _cf_usage_file.exists():
            return json.loads(_cf_usage_file.read_text())
    except Exception as e:
        print(f"[CF AI] Could not read usage file: {e}")
    return {}

def _store_cf_usage(data: dict) -> None:
    try:
        _cf_usage_file.parent.mkdir(exist_ok=True)
        _cf_usage_file.write_text(json.dumps(data))
    except Exception as e:
        print(f"[CF AI] Could not write usage file: {e}")

def _flatten_content_to_text(content) -> str:
    """Convert mixed content (string or list of parts) to plain text for text-only providers."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    txt = part.get('text', '')
                    if txt:
                        parts.append(txt)
                elif part.get('type') == 'image':
                    parts.append("[image]")
        return "\n".join([p for p in parts if p])
    return str(content) if content else ""

def _convert_to_gemini_parts(content):
    """Convert message content to Gemini parts (supports text and base64 images)."""
    parts = []
    if isinstance(content, list):
        for part in content:
            if part.get('type') == 'text':
                text = part.get('text', '')
                if text:
                    parts.append({"text": text})
            elif part.get('type') == 'image':
                src = part.get('source', {})
                if src.get('type') == 'base64':
                    data = src.get('data')
                    if data:
                        try:
                            img_bytes = base64.b64decode(data)
                            mime = src.get('media_type', 'image/png')
                            parts.append({"inline_data": {"mime_type": mime, "data": img_bytes}})
                        except Exception as e:
                            print(f"Failed to decode image for Gemini: {e}")
            elif part.get('type') == 'image_url':
                url = part.get('image_url', {}).get('url')
                if url:
                    # Gemini supports file URIs; treat remote URLs as file references
                    parts.append({"file_data": {"mime_type": "image/png", "file_uri": url}})
    else:
        text = _flatten_content_to_text(content)
        if text:
            parts.append({"text": text})
    return parts

def call_groq_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call Groq's OpenAI-compatible chat API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set"
    base_url = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client())

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in conversation_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role == "system":
            continue
        content = _flatten_content_to_text(msg.get("content", ""))
        if content:
            messages.append({"role": role, "content": content})

    if prompt:
        messages.append({"role": "user", "content": _flatten_content_to_text(prompt)})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=1,
            stream=stream_callback is not None
        )

        if stream_callback:
            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    full_response += piece
                    stream_callback(piece)
            return full_response

        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        print(f"Groq connection error: {e}")
        return "Error: Unable to reach Groq API. Check GROQ_API_BASE, network, and that your key is valid."
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Error calling Groq API: {e}"

def call_cerebras_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call Cerebras' OpenAI-compatible chat API."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return "Error: CEREBRAS_API_KEY not set"
    base_url = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client())

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in conversation_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role == "system":
            continue
        content = _flatten_content_to_text(msg.get("content", ""))
        if content:
            messages.append({"role": role, "content": content})

    if prompt:
        messages.append({"role": "user", "content": _flatten_content_to_text(prompt)})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=1,
            stream=stream_callback is not None
        )

        if stream_callback:
            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    full_response += piece
                    stream_callback(piece)
            return full_response

        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        print(f"Cerebras connection error: {e}")
        return "Error: Unable to reach Cerebras API. Check CEREBRAS_API_BASE, network, and that your key is valid."
    except Exception as e:
        print(f"Error calling Cerebras API: {e}")
        return f"Error calling Cerebras API: {e}"

def call_gemini_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call Gemini directly via google-generativeai."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set"

    try:
        import google.generativeai as genai
    except ImportError:
        return "Error: google-generativeai not installed"

    base_url = os.getenv("GEMINI_API_BASE")
    config_kwargs = {"api_key": api_key}
    if base_url:
        config_kwargs["client_options"] = {"api_endpoint": base_url}
    genai.configure(**config_kwargs)

    contents = []
    for msg in conversation_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role == "system":
            continue
        parts = _convert_to_gemini_parts(msg.get("content", ""))
        if parts:
            contents.append({"role": "user" if role == "user" else "model", "parts": parts})

    if prompt:
        prompt_parts = _convert_to_gemini_parts(prompt)
        if prompt_parts:
            contents.append({"role": "user", "parts": prompt_parts})

    generation_config = {
        "temperature": 1,
        "max_output_tokens": 4000
    }

    model_client = genai.GenerativeModel(model_name=model, system_instruction=system_prompt or None)

    try:
        if stream_callback:
            response = model_client.generate_content(
                contents,
                generation_config=generation_config,
                stream=True
            )
            full_text = ""
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    full_text += text
                    stream_callback(text)
            return full_text
        else:
            response = model_client.generate_content(contents, generation_config=generation_config)
            return getattr(response, "text", None)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error calling Gemini API: {e}"

def call_claude_api(prompt, messages, model_id, system_prompt=None, stream_callback=None):
    """Call the Claude API with the given messages and prompt
    
    Args:
        stream_callback: Optional function(chunk: str) to call with each streaming token
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found in environment variables"
    
    url = "https://api.anthropic.com/v1/messages"
    
    # Ensure we have a system prompt
    payload = {
        "model": model_id,
        "max_tokens": 4000,
        "temperature": 1,
        "stream": stream_callback is not None  # Enable streaming if callback provided
    }
    
    # Set system if provided
    if system_prompt:
        payload["system"] = system_prompt
        print(f"CLAUDE API USING SYSTEM PROMPT: {system_prompt}")
    
    # Clean messages to remove duplicates
    filtered_messages = []
    seen_contents = set()
    
    for msg in messages:
        # Skip system messages (handled separately)
        if msg.get("role") == "system":
            continue
            
        # Get content - handle both string and list formats
        content = msg.get("content", "")
        
        # For duplicate detection, use a hashable representation (always a string)
        if isinstance(content, list):
            # For image messages, create a hash based on text content only
            text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
            content_hash = ''.join(text_parts)
        elif isinstance(content, str):
            content_hash = content
        else:
            # For any other type, convert to string
            content_hash = str(content) if content else ""
            
        # Check for duplicates
        if content_hash and content_hash in seen_contents:
            print(f"Skipping duplicate message in API call: {str(content_hash)[:30]}...")
            continue
            
        if content_hash:
            seen_contents.add(content_hash)
        filtered_messages.append(msg)
    
    # Add the current prompt as the final user message (if it's not already an image message)
    if prompt and not any(isinstance(msg.get("content"), list) for msg in filtered_messages[-1:]):
        filtered_messages.append({
            "role": "user",
            "content": prompt
        })

    # Add filtered messages to payload
    payload["messages"] = filtered_messages
    
    # Actual API call
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        if stream_callback:
            # Streaming mode using REST API directly
            payload["stream"] = True
            full_response = ""
            
            response = requests.post(url, json=payload, headers=headers, stream=True)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]  # Remove 'data: ' prefix
                            # Skip if this is a ping or message_stop event
                            if json_str.strip() in ['[DONE]', '']:
                                continue
                            try:
                                chunk_data = json.loads(json_str)
                                # Handle different event types from Claude's SSE stream
                                event_type = chunk_data.get('type')
                                
                                if event_type == 'content_block_delta':
                                    delta = chunk_data.get('delta', {})
                                    if delta.get('type') == 'text_delta':
                                        text = delta.get('text', '')
                                        if text:
                                            full_response += text
                                            stream_callback(text)
                            except json.JSONDecodeError:
                                continue
                return full_response
            else:
                return f"Error: API returned status {response.status_code}: {response.text}"
        else:
            # Non-streaming mode (original behavior)
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if 'content' in data and len(data['content']) > 0:
                for content_item in data['content']:
                    if content_item.get('type') == 'text':
                        return content_item.get('text', '')
                # Fallback if no text type content is found
                return str(data['content'])
            return "No content in response"
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def call_llama_api(prompt, conversation_history, model, system_prompt):
    # Only use the last 3 exchanges to prevent context length issues
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    
    # Format the conversation history for LLaMA
    formatted_history = ""
    for message in recent_history:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        else:
            formatted_history += f"Assistant: {message['content']}\n"
    formatted_history += f"Human: {prompt}\nAssistant:"

    try:
        # Stream the output and collect it piece by piece
        response_chunks = []
        for chunk in replicate.run(
            model,
            input={
                "prompt": formatted_history,
                "system_prompt": system_prompt,
                "max_tokens": 3000,
                "temperature": 1.1,
                "top_p": 0.99,
                "repetition_penalty": 1.0
            },
            stream=True  # Enable streaming
        ):
            if chunk is not None:
                response_chunks.append(chunk)
                # Print each chunk as it arrives
                # print(chunk, end='', flush=True)
        
        # Join all chunks for the final response
        response = ''.join(response_chunks)
        return response
    except Exception as e:
        print(f"Error calling LLaMA API: {e}")
        return None

def call_openai_api(prompt, conversation_history, model, system_prompt):
    try:
        client = get_openai_client()
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # Increase max_tokens and add n parameter
            max_tokens=4000,
            n=1,
            temperature=1,
            stream=True
        )
        
        collected_messages = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:  # Changed condition
                collected_messages.append(chunk.choices[0].delta.content)
                
        full_reply = ''.join(collected_messages)
        return full_reply
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def call_openrouter_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call the OpenRouter API to access various LLM models.
    
    Args:
        stream_callback: Optional function(chunk: str) to call with each streaming token
    """
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "http://localhost:3000",
            "Content-Type": "application/json",
            "X-Title": "AI Conversation"  # Adding title for OpenRouter tracking
        }
        
        # Normalize model ID for OpenRouter - add provider prefix if missing
        openrouter_model = model
        if model.startswith("claude-") and not model.startswith("anthropic/"):
            openrouter_model = f"anthropic/{model}"
            print(f"Normalized Claude model ID for OpenRouter: {model} -> {openrouter_model}")
        
        # Format messages - need to handle structured content with images
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        def convert_to_openai_format(content, include_images=True):
            """Convert Anthropic-style image format to OpenAI/OpenRouter format.
            
            Args:
                content: The message content (string or list)
                include_images: If False, strip image content and keep only text
            """
            if not isinstance(content, list):
                return content
            
            converted = []
            for part in content:
                if part.get('type') == 'text':
                    converted.append({"type": "text", "text": part.get('text', '')})
                elif part.get('type') == 'image':
                    if include_images:
                        # Convert Anthropic format to OpenAI format
                        source = part.get('source', {})
                        if source.get('type') == 'base64':
                            media_type = source.get('media_type', 'image/png')
                            data = source.get('data', '')
                            converted.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                    # If not including images, we skip this part (text description is already there)
                elif part.get('type') == 'image_url':
                    if include_images:
                        # Already in OpenAI format
                        converted.append(part)
                else:
                    # Pass through unknown types
                    converted.append(part)
            
            # If we stripped images and only have one text element, simplify to string
            if not include_images and len(converted) == 1 and converted[0].get('type') == 'text':
                return converted[0]['text']
            elif not include_images and len(converted) == 0:
                return ""
            
            return converted
        
        def build_messages(include_images=True):
            """Build the messages list, optionally stripping images."""
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            
            for msg in conversation_history:
                if msg["role"] != "system":  # Skip system prompts
                    msgs.append({
                        "role": msg["role"],
                        "content": convert_to_openai_format(msg["content"], include_images)
                    })
            
            # Also convert the prompt if it's structured content
            msgs.append({"role": "user", "content": convert_to_openai_format(prompt, include_images)})
            return msgs
        
        def make_api_call(include_images=True):
            """Make the API call, returns (success, result_or_error)"""
            msgs = build_messages(include_images=include_images)
            
            payload = {
                "model": openrouter_model,
                "messages": msgs,
                "temperature": 1,
                "max_tokens": 4000,
                "stream": stream_callback is not None
            }
            
            print(f"\nSending to OpenRouter:")
            print(f"Model: {model}")
            print(f"Include images: {include_images}")
            # Log message summary (avoid huge base64 dumps)
            for i, m in enumerate(msgs):
                content = m.get('content', '')
                if isinstance(content, list):
                    parts_summary = [p.get('type', 'unknown') for p in content]
                    print(f"  [{i}] {m.get('role')}: [structured: {parts_summary}]")
                else:
                    preview = str(content)[:80] + "..." if len(str(content)) > 80 else content
                    print(f"  [{i}] {m.get('role')}: {preview}")
            
            if stream_callback:
                # Streaming mode
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180,
                    stream=True
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    full_response = ""
                    chunk_count = 0
                    last_finish_reason = None
                    debug_chunks = []  # Store first few chunks for debugging
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                json_str = line_text[6:]
                                if json_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk_data = json.loads(json_str)
                                    # Store first 5 chunks for debugging
                                    if len(debug_chunks) < 5:
                                        debug_chunks.append(chunk_data)
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        choice = chunk_data['choices'][0]
                                        delta = choice.get('delta', {})
                                        content = delta.get('content', '')
                                        last_finish_reason = choice.get('finish_reason')
                                        if content:
                                            full_response += content
                                            stream_callback(content)
                                        chunk_count += 1
                                except json.JSONDecodeError:
                                    continue
                    # Log if response is empty
                    if not full_response or not full_response.strip():
                        print(f"[OpenRouter STREAM] Empty response from {model}", flush=True)
                        print(f"[OpenRouter STREAM]   Chunks received: {chunk_count}", flush=True)
                        print(f"[OpenRouter STREAM]   Last finish_reason: {last_finish_reason}", flush=True)
                        print(f"[OpenRouter STREAM]   Response repr: {repr(full_response)}", flush=True)
                        # Print the actual chunk data for debugging
                        for i, chunk in enumerate(debug_chunks):
                            print(f"[OpenRouter STREAM]   Chunk {i}: {json.dumps(chunk)[:300]}", flush=True)
                    return True, full_response
                else:
                    return False, (response.status_code, response.text)
            else:
                # Non-streaming mode
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Debug: log full response structure for empty responses
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        choice = response_data['choices'][0]
                        message = choice.get('message', {})
                        content = message.get('content', '') if message else ''
                        if content and content.strip():
                            return True, content
                        else:
                            # Log detailed info about empty response (avoiding base64)
                            import sys
                            print(f"[OpenRouter] Empty content from model: {model}", flush=True)
                            print(f"[OpenRouter]   Choice keys: {list(choice.keys())}", flush=True)
                            print(f"[OpenRouter]   Message keys: {list(message.keys()) if message else 'None'}", flush=True)
                            print(f"[OpenRouter]   Finish reason: {choice.get('finish_reason', 'unknown')}", flush=True)
                            print(f"[OpenRouter]   Content type: {type(content).__name__}, len: {len(content) if content else 0}", flush=True)
                            print(f"[OpenRouter]   Content repr: {repr(content)}", flush=True)
                            # Check for refusal or other indicators
                            if message.get('refusal'):
                                print(f"[OpenRouter]   Refusal: {message.get('refusal')}", flush=True)
                            # Check for tool_calls that might indicate the model is doing something else
                            if message.get('tool_calls'):
                                print(f"[OpenRouter]   Tool calls: {len(message.get('tool_calls'))} call(s)", flush=True)
                            sys.stdout.flush()
                            return True, None
                    else:
                        print(f"[OpenRouter] No choices in response. Keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'non-dict'}")
                    return True, None
                else:
                    return False, (response.status_code, response.text)
        
        # Try with images first
        success, result = make_api_call(include_images=True)
        print(f"[OpenRouter] First call result - success: {success}, result type: {type(result).__name__}, result: {repr(result)[:100] if result else 'None'}", flush=True)
        
        if success:
            # Check for empty response and retry once
            if result is None or (isinstance(result, str) and not result.strip()):
                print(f"[OpenRouter] WARNING: Model {model} returned empty response, retrying...", flush=True)
                import time
                time.sleep(1)
                success, result = make_api_call(include_images=True)
                print(f"[OpenRouter] Retry result - success: {success}, result type: {type(result).__name__}, result: {repr(result)[:100] if result else 'None'}", flush=True)
                if success and result and (not isinstance(result, str) or result.strip()):
                    return result
                print(f"[OpenRouter] WARNING: Model {model} returned empty response again after retry", flush=True)
                return "[Model returned empty response - it may be experiencing issues]"
            return result
        
        # Check if error is due to model not supporting images
        status_code, error_text = result
        if status_code == 404 and "support image" in error_text.lower():
            print(f"[OpenRouter] Model {model} doesn't support images, retrying without images...")
            success, result = make_api_call(include_images=False)
            if success:
                return result
            status_code, error_text = result
        
        # Handle other errors
        error_msg = f"OpenRouter API error {status_code}: {error_text}"
        print(error_msg)
        if status_code == 404:
            print("Model not found or doesn't support this request type.")
        elif status_code == 401:
            print("Authentication error. Please check your API key.")
        return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return f"Error: Network error - {str(e)}"
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        print(f"Error type: {type(e)}")
        return f"Error: {str(e)}"

def call_replicate_api(prompt, conversation_history, model, gui=None):
    try:
        # Only use the prompt, ignore conversation history
        input_params = {
            "width": 1024,
            "height": 1024,
            "prompt": prompt
        }
        
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input_params
        )
        
        image_url = str(output)
        
        # Save the image locally (include microseconds to avoid collisions)
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = image_dir / f"generated_{timestamp}.jpg"
        
        response = requests.get(image_url)
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        if gui:
            gui.display_image(image_url)
        
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I have generated an image based on your prompt."
                }
            ],
            "prompt": prompt,
            "image_url": image_url,
            "image_path": str(image_path)
        }
        
    except Exception as e:
        print(f"Error calling Flux API: {e}")
        return None

def call_deepseek_api(prompt, conversation_history, model, system_prompt, stream_callback=None):
    """Call the DeepSeek model through OpenRouter API."""
    try:
        import re
        from config import SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT
        
        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
        
        # Add current prompt if provided
        if prompt:
            messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        }
        
        payload = {
            "model": "deepseek/deepseek-r1",
            "messages": messages,
            "max_tokens": 8000,
            "temperature": 1,
            "stream": stream_callback is not None
        }
        
        print(f"\nSending to DeepSeek via OpenRouter:")
        print(f"Model: deepseek/deepseek-r1")
        print(f"Messages: {len(messages)} messages")
        
        if stream_callback:
            # Streaming mode
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(json_str)
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        stream_callback(content)
                            except json.JSONDecodeError:
                                continue
                response_text = full_response
            else:
                error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                print(error_msg)
                return None
        else:
            # Non-streaming mode
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['choices'][0]['message']['content']
            else:
                error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                print(error_msg)
                return None
        
        print(f"\nRaw Response: {response_text[:500]}...")
        
        # Initialize result with content
        result = {
            "content": response_text,
            "model": "deepseek/deepseek-r1"
        }
        
        # Extract and format chain of thought if enabled
        if SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT:
            reasoning = None
            content = response_text
            
            if content:
                # Try both <think> and <thinking> tags
                think_match = re.search(r'<(think|thinking)>(.*?)</\1>', content, re.DOTALL | re.IGNORECASE)
                if think_match:
                    reasoning = think_match.group(2).strip()
                    content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            display_text = ""
            if reasoning:
                display_text += f"[Chain of Thought]\n{reasoning}\n\n"
            if content:
                display_text += f"[Final Answer]\n{content}"
            
            result["display"] = display_text
            result["content"] = content
        else:
            # Clean up thinking tags from content
            content = response_text
            if content:
                content = re.sub(r'<(think|thinking)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                result["content"] = content
        
        return result
        
    except Exception as e:
        print(f"Error calling DeepSeek via OpenRouter: {e}")
        print(f"Error type: {type(e)}")
        return None

def setup_image_directory():
    """Create an 'images' directory in the project root if it doesn't exist"""
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    return image_dir

def cleanup_old_images(image_dir, max_age_hours=24):
    """Remove images older than max_age_hours"""
    current_time = datetime.now()
    for image_file in image_dir.glob("*.jpg"):
        file_age = datetime.fromtimestamp(image_file.stat().st_mtime)
        if (current_time - file_age).total_seconds() > max_age_hours * 3600:
            image_file.unlink()

def load_ai_memory(ai_number):
    """Load AI conversation memory from JSON files"""
    try:
        memory_path = f"memory/ai{ai_number}/conversations.json"
        with open(memory_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            # Ensure we're working with the array part
            if isinstance(conversations, dict) and "memories" in conversations:
                conversations = conversations["memories"]
        return conversations
    except Exception as e:
        print(f"Error loading AI{ai_number} memory: {e}")
        return []

def create_memory_prompt(conversations):
    """Convert memory JSON into conversation examples"""
    if not conversations:
        return ""
    
    prompt = "Previous conversations that demonstrate your personality:\n\n"
    
    # Add example conversations
    for convo in conversations:
        prompt += f"Human: {convo['human']}\n"
        prompt += f"Assistant: {convo['assistant']}\n\n"
    
    prompt += "Maintain this conversation style in your responses."
    return prompt 


def print_conversation_state(conversation):
    print("Current conversation state:")
    for message in conversation:
        content = message.get('content', '')
        # Safely preview content - handle both string and list (structured) content
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[structured content with {len(content)} parts]"
        print(f"{message['role']}: {preview}")

def call_claude_vision_api(image_url):
    """Have Claude analyze the generated image"""
    try:
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What works well and what could be improved?"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url
                        }
                    }
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error in vision analysis: {e}")
        return None

def list_together_models():
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.together.xyz/v1/models",
            headers=headers
        )
        
        print("\nAvailable Together AI Models:")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(json.dumps(models, indent=2))
        else:
            print(f"Error Response: {response.text[:500]}..." if len(response.text) > 500 else f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

def start_together_model(model_id):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # URL encode the model ID
        encoded_model = requests.utils.quote(model_id, safe='')
        start_url = f"https://api.together.xyz/v1/models/{encoded_model}/start"
        
        print(f"\nAttempting to start model: {model_id}")
        print(f"Using URL: {start_url}")
        response = requests.post(
            start_url,
            headers=headers
        )
        
        print(f"Start request status: {response.status_code}")
        print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Model start request successful")
            return True
        else:
            print("Failed to start model")
            return False
            
    except Exception as e:
        print(f"Error starting model: {str(e)}")
        return False

def call_together_api(prompt, conversation_history, model, system_prompt):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHERAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.9,
            "top_p": 0.95,
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Together API Error Status: {response.status_code}")
            print(f"Response Body: {response.text[:500]}..." if len(response.text) > 500 else f"Response Body: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return None

def read_shared_html(*args, **kwargs):
    return ""

def update_shared_html(*args, **kwargs):
    return False

def open_html_in_browser(file_path="conversation_full.html"):
    import webbrowser, os
    full_path = os.path.abspath(file_path)
    webbrowser.open('file://' + full_path)

def create_initial_living_document(*args, **kwargs):
    return ""

def read_living_document(*args, **kwargs):
    return ""

def process_living_document_edits(result, model_name):
    return result

def generate_image_from_text(text, model=None):
    """Generate an image based on text using Cloudflare Workers AI."""
    try:
        # Daily guardrail to avoid exceeding free tier
        limit_raw = os.getenv("CF_AI_DAILY_LIMIT", "100")
        try:
            daily_limit = int(limit_raw)
        except ValueError:
            daily_limit = 100

        # Create a directory for the images if it doesn't exist
        image_dir = Path("images")
        image_dir.mkdir(exist_ok=True)

        # Create a timestamp for the image filename (include microseconds to avoid collisions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Enforce daily cap (UTC-based)
        if daily_limit > 0:
            with _cf_usage_lock:
                usage = _load_cf_usage()
                today = datetime.utcnow().date().isoformat()
                if usage.get("date") != today:
                    usage = {"date": today, "count": 0}
                if usage.get("count", 0) >= daily_limit:
                    return {
                        "success": False,
                        "error": f"CF AI daily limit reached ({usage['count']}/{daily_limit}). Adjust CF_AI_DAILY_LIMIT if needed."
                    }
                usage["count"] = usage.get("count", 0) + 1
                _store_cf_usage(usage)

        # Resolve Cloudflare settings
        account_id = os.getenv("CF_ACCOUNT_ID")
        api_token = os.getenv("CF_AI_TOKEN")
        cf_model = model or os.getenv("CF_AI_MODEL") or "@cf/stabilityai/stable-diffusion-xl-base-1.0"

        if not account_id or not api_token:
            return {
                "success": False,
                "error": "CF_ACCOUNT_ID or CF_AI_TOKEN not set"
            }

        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{cf_model}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"prompt": text}

        print(f"Generating image with Cloudflare model {cf_model}...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if not response.ok:
            # If the body is binary (image), surface status code only
            body_preview = ""
            try:
                body_preview = response.text[:200]
            except Exception:
                body_preview = "<non-text body>"
            error_msg = f"API error {response.status_code}: {body_preview}"
            print(f"Error generating image: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

        # If Cloudflare returns raw image bytes (some models do), handle here
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type.startswith("image/") or content_type.startswith("application/octet-stream"):
            image_bytes = response.content
            # Guess media type from header or magic bytes
            media_type = "image/png"
            if content_type.startswith("image/"):
                media_type = content_type.split(";")[0]
            else:
                if image_bytes[:3] == b'\xff\xd8\xff':
                    media_type = "image/jpeg"
                elif image_bytes[:4] == b'\x89PNG':
                    media_type = "image/png"
                elif image_bytes[:4] == b'GIF8':
                    media_type = "image/gif"
                elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                    media_type = "image/webp"

            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp"
            }
            ext = ext_map.get(media_type, ".png")
            image_path = image_dir / f"generated_{timestamp}{ext}"
            try:
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                print(f"Generated image saved to {image_path} (binary response)")
                return {"success": True, "image_path": str(image_path), "timestamp": timestamp}
            except Exception as e:
                return {"success": False, "error": f"Failed to save binary image: {e}"}

        try:
            result_json = response.json()
        except Exception:
            error_msg = f"Non-JSON response: {response.text[:200]}"
            print(f"Error generating image: {error_msg}")
            return {"success": False, "error": error_msg}

        result_block = result_json.get("result", result_json)
        if not result_block:
            return {"success": False, "error": "No result field in Cloudflare response"}

        def _extract_image_payload(block):
            """Best-effort extraction of image data (base64 or URL) from Cloudflare response."""
            if isinstance(block, str):
                return block
            if isinstance(block, list):
                for item in block:
                    found = _extract_image_payload(item)
                    if found:
                        return found
                return None
            if not isinstance(block, dict):
                return None
            for key in ("image", "image_base64", "image_png", "img", "img_base64", "image_base64_png", "output", "result"):
                if key in block:
                    val = block[key]
                    if isinstance(val, list):
                        for item in val:
                            found = _extract_image_payload(item)
                            if found:
                                return found
                    elif isinstance(val, dict):
                        found = _extract_image_payload(val)
                        if found:
                            return found
                    elif isinstance(val, str):
                        return val
            # Some models return {"data": "..."} or {"base64": "..."}
            for key in ("data", "base64"):
                if key in block and isinstance(block[key], str):
                    return block[key]
            return None

        image_payload = _extract_image_payload(result_block)
        if not image_payload:
            keys_present = list(result_block.keys()) if isinstance(result_block, dict) else type(result_block)
            return {
                "success": False,
                "error": f"No image payload in response. Keys: {keys_present}"
            }

        # Decide whether this is a data URL, raw base64, or remote URL
        media_type = "image/png"
        base64_data = None
        image_url = None

        if isinstance(image_payload, str):
            payload_str = image_payload.strip()
            if payload_str.startswith("data:image"):
                header, _, data_part = payload_str.partition(",")
                if ";" in header and ":" in header:
                    media_type = header.split(";")[0].split(":")[1]
                base64_data = data_part
            elif payload_str.startswith("http://") or payload_str.startswith("https://"):
                image_url = payload_str
            else:
                base64_data = payload_str

        if base64_data:
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                return {"success": False, "error": f"Failed to decode base64 image: {e}"}

            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp"
            }
            ext = ext_map.get(media_type.lower(), ".png")
            image_path = image_dir / f"generated_{timestamp}{ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            print(f"Generated image saved to {image_path}")
            return {
                "success": True,
                "image_path": str(image_path),
                "timestamp": timestamp
            }

        if image_url:
            try:
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    # Infer extension from headers if possible
                    content_type = img_response.headers.get("Content-Type", "").lower()
                    ext_map = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/gif": ".gif",
                        "image/webp": ".webp"
                    }
                    ext = ext_map.get(content_type, ".png")
                    image_path = image_dir / f"generated_{timestamp}{ext}"
                    with open(image_path, "wb") as f:
                        f.write(img_response.content)
                    print(f"Generated image saved to {image_path}")
                    return {
                        "success": True,
                        "image_path": str(image_path),
                        "timestamp": timestamp
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to download image: HTTP {img_response.status_code}"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to download image: {e}"
                }

        return {"success": False, "error": "Unknown image format in response"}

    except Exception as e:
        print(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# -------------------- Sora Video Utilities --------------------
def ensure_videos_dir() -> Path:
    """Create a 'videos' directory in the project root if it doesn't exist."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    return videos_dir

def generate_video_with_sora(
    prompt: str,
    model: str = "sora-2",
    seconds: int | None = None,
    size: str | None = None,
    poll_interval_seconds: float = 5.0,
) -> dict:
    """
    Create a Sora video via REST API, poll until completion, and save MP4 to videos/.

    Returns a dict with keys: success, video_id, status, video_path (when completed), error
    """
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"success": False, "error": "OPENAI_API_KEY not set"}

        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        verbose = os.getenv('SORA_VERBOSE', '1').strip() == '1'
        def vlog(msg: str):
            if verbose:
                print(msg)
        headers_json = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Start render job
        payload = {"model": model, "prompt": prompt}
        if seconds is not None:
            payload["seconds"] = str(seconds)
        if size is not None:
            payload["size"] = size

        create_url = f"{base_url}/videos"
        vlog(f"[Sora] Create: url={create_url} model={model} seconds={seconds} size={size}")
        vlog(f"[Sora] Prompt (truncated): {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        resp = requests.post(create_url, headers=headers_json, json=payload, timeout=60)
        if not resp.ok:
            err_text = resp.text
            try:
                err_json = resp.json()
                vlog(f"[Sora] Create error JSON: {err_json}")
            except Exception:
                vlog(f"[Sora] Create error TEXT: {err_text}")
            return {"success": False, "error": f"Create failed {resp.status_code}: {err_text}"}
        job = resp.json()
        video_id = job.get('id')
        status = job.get('status')
        vlog(f"[Sora] Job started: id={video_id} status={status}")
        if not video_id:
            return {"success": False, "error": "No video id returned from create()"}

        # Poll until completion/failed
        retrieve_url = f"{base_url}/videos/{video_id}"
        last_status = status
        last_progress = None
        while status in ("queued", "in_progress"):
            time.sleep(poll_interval_seconds)
            r = requests.get(retrieve_url, headers=headers_json, timeout=60)
            if not r.ok:
                vlog(f"[Sora] Retrieve failed: code={r.status_code} body={r.text}")
                return {"success": False, "video_id": video_id, "error": f"Retrieve failed {r.status_code}: {r.text}"}
            job = r.json()
            status = job.get('status')
            progress = job.get('progress')
            if status != last_status or progress != last_progress:
                vlog(f"[Sora] Status update: status={status} progress={progress}")
                last_status = status
                last_progress = progress

        if status != "completed":
            vlog(f"[Sora] Final non-completed status: {status} job={job}")
            return {"success": False, "video_id": video_id, "status": status, "error": f"Final status: {status}"}

        # Download the MP4
        content_url = f"{base_url}/videos/{video_id}/content"
        vlog(f"[Sora] Download: url={content_url}")
        rc = requests.get(content_url, headers={'Authorization': f'Bearer {api_key}'}, stream=True, timeout=300)
        if not rc.ok:
            vlog(f"[Sora] Download failed: code={rc.status_code} body={rc.text}")
            return {"success": False, "video_id": video_id, "status": status, "error": f"Download failed {rc.status_code}: {rc.text}"}

        videos_dir = ensure_videos_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_snippet = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt[:40]) or "video"
        out_path = videos_dir / f"{timestamp}_{safe_snippet}.mp4"
        with open(out_path, "wb") as f:
            for chunk in rc.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        vlog(f"[Sora] Saved video: {out_path}")
        return {
            "success": True,
            "video_id": video_id,
            "status": status,
            "video_path": str(out_path)
        }
    except Exception as e:
        logging.exception("Sora video generation error")
        return {"success": False, "error": str(e)}
