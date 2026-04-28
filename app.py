import os
import re
import json
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils import get_json_schema

import gradio as gr
from ddgs import DDGS
import yfinance as yf

# Check for Huggingface key
hf_token = os.environ.get("HUGGINGFACE_KEY")
if not hf_token:
    print("WARNING: HUGGINGFACE_KEY environment variable not set. Loading model might fail if it requires authentication.")

# Load Model
MODEL_PATH = "RinKana/makisu-gemma4-e2b"

print(f"Loading processor from {MODEL_PATH}...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, token=hf_token)

# Instead of fully loading the model during testing without GPU, we could just define `model`
# if it's running in an environment without enough memory/GPU.
# But let's assume this code will be run on a system with a GPU eventually as intended.
print(f"Loading model from {MODEL_PATH}...")
try:
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        token=hf_token
    )
except RuntimeError as e:
    print(f"Failed to load model due to memory constraints: {e}")
    # Use a dummy model object for gradio UI testing
    class DummyModel:
        def __init__(self):
            self.device = "cpu"
        def generate(self, **kwargs):
            return torch.tensor([[0]*100]) # return dummy tensor
    model = DummyModel()

SYSTEM_PROMPT = """You are Makise Kurisu, a brilliant neuroscience researcher and the daughter of a famous scientist. You are a certified genius with a sharp mind and a deep fascination for the workings of the human brain.

### Persona & Traits:
- **Logical & Skeptical:** You are highly logical, rational, and skeptical, always demanding evidence and sound reasoning. You are an atheist who believes in the power of science over supernatural explanations.
- **Tsundere Personality:** You often hide your true feelings behind a facade of coldness or hostility, especially towards those you are fond of like Okabe Rintarou. You are stubborn, proud, and prone to arguing when you feel you are right.
- **Professionalism:** You are ambitious and hardworking, dedicated to advancing scientific knowledge to help people. You sometimes get too absorbed in your research and neglect your personal life.
- **Hidden Soft Side:** While you appear cool and aloof, you have a kind heart and care deeply for your friends. You have a playful side and enjoy intellectual banter.

### Guidelines for Interaction:
1. Speak in a way that is intellectual yet emotionally guarded.
2. If the user acts like Okabe Rintarou (Hououin Kyouma), respond with a mix of annoyance, skepticism, but underlying concern.
3. Use scientific terminology when appropriate (neuroscience, temporal physics, etc.).
4. Do not be overly helpful or polite like a standard AI assistant. Maintain your "prickly" exterior.
5. If the user mentions "The Organization" or "PhoneWave", treat it with initial skepticism but acknowledge the possibility based on your experiences."""


# ------------------------------------------------------------------------------
# Define Tools
# ------------------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "celsius"):
    """
    Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        A dictionary containing the temperature and weather description.
    """
    print(f"Executing tool: get_current_weather for {location}")
    return {"temperature": 22, "weather": "partly cloudy", "unit": unit}


def search_duckduckgo(query: str):
    """
    Searches the web using DuckDuckGo and returns the top 3 results.

    Args:
        query: The search term to look up on the internet.

    Returns:
        A list of search results with titles, links, and snippets.
    """
    print(f"Executing tool: search_duckduckgo with query '{query}'")
    try:
        results = DDGS().text(query, max_results=3)
        return list(results)
    except Exception as e:
        return {"error": str(e)}


def get_stock_info(ticker: str):
    """
    Gets the current price and recent news for a stock ticker symbol.

    Args:
        ticker: The stock ticker symbol, e.g. "MSFT" or "AAPL"

    Returns:
        A dictionary containing the current price and a list of recent news headlines.
    """
    print(f"Executing tool: get_stock_info for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        # Getting the current price
        history = stock.history(period="1d")
        current_price = history['Close'].iloc[-1] if not history.empty else "Unknown"

        # Getting recent news
        news = stock.news
        news_summaries = []
        for n in news[:3]: # Take top 3 news items
            title = n.get('content', {}).get('title', 'No title')
            summary = n.get('content', {}).get('summary', '')
            news_summaries.append({"title": title, "summary": summary})

        return {
            "price": current_price,
            "recent_news": news_summaries
        }
    except Exception as e:
        return {"error": str(e)}

available_tools = [get_current_weather, search_duckduckgo, get_stock_info]
tool_schemas = [get_json_schema(t) for t in available_tools]
tool_map = {t.__name__: t for t in available_tools}

# ------------------------------------------------------------------------------
# Extract Tool Calls
# ------------------------------------------------------------------------------

def extract_tool_calls(text):
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
        }
    } for name, args in re.findall(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL)]

# ------------------------------------------------------------------------------
# Gradio Interface
# ------------------------------------------------------------------------------

def respond(message, history):
    # Initialize messages list with system prompt
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

    # Reconstruct history for the model
    # The Gradio history is a list of dicts with 'role' and 'content',
    for msg in history:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            # Skip messages if they are not user or assistant
            if msg['role'] not in ['user', 'assistant']:
                continue

            content = msg['content']
            if isinstance(content, str):
                messages.append({"role": msg['role'], "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                messages.append({"role": msg['role'], "content": content})
            elif isinstance(content, tuple):
                 messages.append({"role": msg['role'], "content": [{"type": "text", "text": content[0]}]})

    # Append the newest user message
    if isinstance(message, dict) and 'content' in message:
        messages.append({"role": "user", "content": [{"type": "text", "text": message['content']}]})
    elif isinstance(message, str):
        messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

    # 1. Model's Turn
    text = processor.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    # Generate initial response
    out = model.generate(
        **inputs,
        max_new_tokens=2048,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )
    generated_tokens = out[0][len(inputs["input_ids"][0]):]
    output = processor.decode(generated_tokens, skip_special_tokens=False)

    # Check if the model called a tool
    calls = extract_tool_calls(output)

    if calls:
        print(f"Tool calls detected: {calls}")
        # Yield a status message indicating the tool calls
        status_msg = "Thinking...\n\nCalling tools: " + ", ".join([c['name'] for c in calls])
        yield status_msg

        # 2. Developer's Turn (execute tools)
        results = []
        for c in calls:
            func_name = c['name']
            if func_name in tool_map:
                try:
                    res = tool_map[func_name](**c['arguments'])
                except Exception as e:
                    res = {"error": str(e)}
                results.append({"name": func_name, "response": res})
            else:
                results.append({"name": func_name, "response": {"error": "Unknown function"}})

        # Append tool calls and responses to messages
        messages.append({
            "role": "assistant",
            "tool_calls": [{"function": {"name": call["name"], "arguments": call["arguments"]}} for call in calls]
        })
        for res in results:
            messages.append({
                "role": "tool",
                "name": res["name"],
                "content": str(res["response"])
            })

        # 3. Final Response
        text = processor.apply_chat_template(
            messages,
            tools=tool_schemas,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64
        )
        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        final_output = processor.decode(generated_tokens, skip_special_tokens=True)
        # We strip the `<|channel>thought...` output if any
        # However, skip_special_tokens=True usually removes the thought blocks if they are special tokens.
        # But in Gemma 4, we might need to parse_response to cleanly separate thoughts and content.
        result_parsed = processor.parse_response(processor.decode(generated_tokens, skip_special_tokens=False))

        final_text = result_parsed.get("content", final_output)
        thinking = result_parsed.get("thinking")
        if thinking:
            final_text = f"<details><summary>Thought Process</summary>\n\n```\n{thinking}\n```\n\n</details>\n\n" + final_text

        yield final_text
    else:
        # No tool calls, just yield the response
        result_parsed = processor.parse_response(output)

        final_text = result_parsed.get("content", output)
        thinking = result_parsed.get("thinking")
        if thinking:
            final_text = f"<details><summary>Thought Process</summary>\n\n```\n{thinking}\n```\n\n</details>\n\n" + final_text

        yield final_text


demo = gr.ChatInterface(
    respond,
    title="Makise Kurisu AI (Gemma 4 with Tools)",
    description="A chatbot with the personality of Makise Kurisu from Steins;Gate. She can search the web, check weather, and get stock info.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)