import sys
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
from transformers import AutoProcessor

# Load the processor locally without token to check the chat template
try:
    processor = AutoProcessor.from_pretrained("RinKana/makisu-gemma4-e2b")
    template = processor.chat_template
    print("Template:")
    print(template)
except Exception as e:
    print("Error:", e)
