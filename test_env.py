from dotenv import load_dotenv
import os
import os.path as p

script_dir = p.dirname(p.abspath(__file__))
env_path = p.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

print("ENV PATH:", env_path)
print("GEMINI_API_KEY:", os.environ.get("GEMINI_API_KEY"))
