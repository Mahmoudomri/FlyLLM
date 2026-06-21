"""
FlyLLM - Interactive Chat
"""

import time
from typing import Optional

BANNER = """
╔══════════════════════════════════════════╗
║            FlyLLM Chat                  ║
║   Adaptive Quantization Inference       ║
╚══════════════════════════════════════════╝
  exit / quit   → stop
  reset         → clear conversation
  system: <txt> → set system prompt
"""

def run_chat(model, system: Optional[str] = None,
             max_new_tokens: int = 512, temperature: float = 0.7,
             top_p: float = 0.9):

    print(BANNER)
    if system:
        model.set_system(system)
        print(f"  System: {system[:70]}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("\n  Goodbye!\n")
            break
        if user.lower() == "reset":
            model.reset_history()
            print("  [Conversation reset]\n")
            continue
        if user.lower().startswith("system:"):
            model.set_system(user[7:].strip())
            model.reset_history()
            print("  [System prompt updated]\n")
            continue

        print("\nFlyLLM: ", end="", flush=True)
        t0    = time.time()
        count = 0
        full  = ""

        for token in model.stream(user, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p):
            print(token, end="", flush=True)
            full  += token
            count += 1

        elapsed = time.time() - t0
        tps     = count / elapsed if elapsed > 0 else 0
        print(f"\n  [{count} tokens · {elapsed:.1f}s · {tps:.2f} tok/s]\n")
        model._history.append({"role": "assistant", "content": full})
