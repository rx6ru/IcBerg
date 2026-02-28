import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from backend.main import app

load_dotenv()

session_id = f"test-crash-{int(time.time())}"

queries = [
    "How many passengers embarked from each port?",
    "How many passengers embarked from each port? visualise it",
    "use some chart",
    "give the previous bar graph"
]

with TestClient(app) as client:
    for query in queries:
        print(f"\n--- REQUEST: {query} ---")
        start = time.time()
        try:
            with client.stream("POST", "/chat/stream", json={"session_id": session_id, "message": query}) as response:
                for line in response.iter_lines():
                    if line:
                        decoded = line
                        if decoded.startswith("data: "):
                            data = json.loads(decoded[6:])
                            event_type = data.get("type")
                            if event_type == "tool_start":
                                print(f"  [TOOL RUNNING] {data.get('name')}")
                            elif event_type == "tool_end":
                                print(f"  [TOOL FINISHED] {data.get('name')}")
                            elif event_type == "final_text":
                                print(f"  [TEXT] {data.get('content')[:100]}...")
                            elif event_type == "image":
                                print(f"  [IMAGE] returned (length: {len(data.get('content'))})")
        except Exception:
            import traceback
            traceback.print_exc()
        print(f"Time: {time.time() - start:.2f}s")
