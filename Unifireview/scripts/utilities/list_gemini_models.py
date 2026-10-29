"""List available Google Gemini models."""
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# Set API key
# GOOGLE_API_KEY must be set in the environment

import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Fetching available Google Gemini models...")
print("=" * 80)

try:
    models = genai.list_models()
    print(f"\nAvailable models:\n")
    for model in models:
        print(f"- {model.name}")
        if hasattr(model, 'display_name'):
            print(f"  Display Name: {model.display_name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Supported Methods: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")
    import traceback
    traceback.print_exc()
