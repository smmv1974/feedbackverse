"""List available Groq models."""
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# Set API key
# GROQ_API_KEY must be set in the environment

from groq import Groq

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

print("Fetching available Groq models...")
print("=" * 80)

try:
    models = client.models.list()
    print(f"\nFound {len(models.data)} models:\n")
    for model in models.data:
        print(f"- {model.id}")
        if hasattr(model, 'owned_by'):
            print(f"  Owner: {model.owned_by}")
        if hasattr(model, 'created'):
            print(f"  Created: {model.created}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")
    import traceback
    traceback.print_exc()
