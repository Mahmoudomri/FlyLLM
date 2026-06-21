"""
FlyLLM - CLI
flyllm run   <model_id_or_path> --prompt "..."
flyllm chat  <model_id_or_path>
flyllm quantize <model_id>
flyllm profile  <model_id>
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="flyllm",
        description="FlyLLM — Adaptive quantization for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p = sub.add_parser("run", help="Run a prompt (auto-downloads and quantizes if needed)")
    p.add_argument("model")
    p.add_argument("--prompt",      "-p", required=True)
    p.add_argument("--system",      "-s", default=None)
    p.add_argument("--max-tokens",  "-m", default=40,  type=int)
    p.add_argument("--temperature", "-t", default=0.7,  type=float)
    p.add_argument("--top-p",             default=0.9,  type=float)

    # chat
    p = sub.add_parser("chat", help="Interactive chat (auto-downloads and quantizes if needed)")
    p.add_argument("model")
    p.add_argument("--system",      "-s", default=None)
    p.add_argument("--max-tokens",  "-m", default=40,  type=int)
    p.add_argument("--temperature", "-t", default=0.7,  type=float)
    p.add_argument("--top-p",             default=0.9,  type=float)

    # quantize
    p = sub.add_parser("quantize", help="Profile and quantize a model")
    p.add_argument("model")

    # profile
    p = sub.add_parser("profile", help="Profile layer sensitivity only")
    p.add_argument("model")
    p.add_argument("--output", "-o", default=None)

    args = parser.parse_args()

    if args.command == "profile":
        from .config import get_hf_cache_dir
        from .profiler import profile_model

        hf_dir = get_hf_cache_dir(args.model)
        if not hf_dir:
            print(f"  Model not in HF cache. Run: flyllm run {args.model} first.")
            sys.exit(1)
        profile_model(args.model, hf_dir=hf_dir,
                      output_path=args.output, verbose=True)

    elif args.command == "quantize":
        from .loader import FlyLLM
        # from_pretrained handles everything
        FlyLLM.from_pretrained(args.model, verbose=True)
        print("  Done. Run: flyllm chat", args.model)

    elif args.command in ("run", "chat"):
        from .loader import FlyLLM

        # Check if it's a direct flyllm dir path
        meta = os.path.join(os.path.expanduser(args.model), "flyllm_meta.json")
        if os.path.exists(meta):
            model = FlyLLM.load(args.model)
        else:
            model = FlyLLM.from_pretrained(args.model)

        if args.command == "run":
            resp = model.generate(
                args.prompt,
                system=args.system,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\n{resp}\n")
        else:
            from .chat import run_chat
            run_chat(model, system=args.system,
                     max_new_tokens=args.max_tokens,
                     temperature=args.temperature,
                     top_p=args.top_p)


if __name__ == "__main__":
    main()
