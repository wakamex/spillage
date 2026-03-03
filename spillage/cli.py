"""CLI entry point for Min-Spill Search."""

from __future__ import annotations

import json
import sys

import click

from .backend_mock import MockBackend
from .config import MSSConfig, get_preset
from .sampler import GenerateResult, TokenEvent, generate, generate_greedy


def _format_event(event: TokenEvent) -> str:
    """Format a TokenEvent for verbose output."""
    parts = [f"[step {event.step}]"]
    if event.fast_path:
        c = event.candidates[0]
        parts.append(f"FAST → {c.token_text!r} (logp={c.log_prob:.3f})")
    else:
        for i, c in enumerate(event.candidates):
            marker = ">>>" if i == event.selected else "   "
            parts.append(
                f"  {marker} {c.token_text!r:10s} "
                f"logp={c.log_prob:.3f}  "
                f"spill={c.spill_raw:.3f}  "
                f"norm={c.spill_normalized:.3f}  "
                f"score={c.score:.3f}"
            )
    if event.panic:
        parts.append("  !! PANIC fallback")
    parts.append(f"  ({event.wall_time_ms:.1f}ms)")
    return "\n".join(parts)


def _event_to_dict(event: TokenEvent) -> dict:
    return {
        "step": event.step,
        "fast_path": event.fast_path,
        "panic": event.panic,
        "wall_time_ms": event.wall_time_ms,
        "selected": event.selected,
        "candidates": [
            {
                "token_id": c.token_id,
                "token_text": c.token_text,
                "log_prob": c.log_prob,
                "spill_raw": c.spill_raw,
                "spill_normalized": c.spill_normalized,
                "score": c.score,
            }
            for c in event.candidates
        ],
    }


@click.group()
def main():
    """Min-Spill Search — lookahead re-ranking for reduced hallucination."""


@main.command()
@click.option("--prompt", "-p", required=True, help="Input prompt text.")
@click.option("--k", default=3, show_default=True, help="Number of lookahead candidates.")
@click.option("--beta", default=2.0, show_default=True, help="Spill penalty weight.")
@click.option("--tau", default=1.0, show_default=True, help="Spill threshold (post-normalization).")
@click.option("--variant", type=click.Choice(["thresholded", "raw"]), default="thresholded", show_default=True)
@click.option("--max-tokens", default=256, show_default=True)
@click.option("--temperature", default=0.0, show_default=True)
@click.option("--adaptive/--no-adaptive", default=True, show_default=True)
@click.option("--model-preset", type=click.Choice(["qwen35-35b", "qwen35-9b", "default"]), default="default")
@click.option("--verbose", "-v", is_flag=True, help="Print per-token diagnostics to stderr.")
@click.option("--json-trace", type=click.Path(), default=None, help="Write full token trace as JSON.")
@click.option("--mock", is_flag=True, hidden=True, help="Use MockBackend (for testing).")
@click.option("--model", type=click.Path(exists=True), envvar="SPILLAGE_MODEL", default=None, help="Path to GGUF model file.")
@click.option("--n-ctx", default=2048, show_default=True, help="Context window size.")
@click.option("--n-gpu-layers", default=-1, show_default=True, help="GPU layers (-1=all).")
def gen(
    prompt: str,
    k: int,
    beta: float,
    tau: float,
    variant: str,
    max_tokens: int,
    temperature: float,
    adaptive: bool,
    model_preset: str,
    verbose: bool,
    json_trace: str | None,
    mock: bool,
    model: str | None,
    n_ctx: int,
    n_gpu_layers: int,
) -> None:
    """Generate text using Min-Spill Search decoding."""
    preset = get_preset(model_preset)
    cfg = MSSConfig(
        k=k,
        beta=beta if beta != 2.0 else preset.beta,
        tau=tau if tau != 1.0 else preset.tau,
        max_tokens=max_tokens,
        temperature=temperature,
        adaptive=adaptive,
        confidence_threshold=preset.confidence_threshold,
        panic_margin=preset.panic_margin,
    )

    if mock:
        backend = MockBackend(vocab_size=16)
    elif model:
        from .backend_native import NativeBackend
        backend = NativeBackend(
            model_path=model, n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers, verbose=verbose,
        )
    else:
        click.echo(
            "Error: provide --model /path/to/model.gguf or --mock.\n"
            "You can also set SPILLAGE_MODEL env var.",
            err=True,
        )
        raise SystemExit(1)

    def on_token(event: TokenEvent) -> None:
        if verbose:
            click.echo(_format_event(event), err=True)
        # Stream the selected token to stdout.
        text = event.candidates[event.selected].token_text
        click.echo(text, nl=False)

    result = generate(
        prompt, backend, config=cfg, variant=variant, on_token=on_token,
    )
    click.echo()  # final newline

    if json_trace:
        trace = {
            "prompt": prompt,
            "config": {
                "k": cfg.k, "beta": cfg.beta, "tau": cfg.tau,
                "variant": variant, "adaptive": cfg.adaptive,
                "temperature": cfg.temperature,
            },
            "total_time_ms": result.total_time_ms,
            "token_count": len(result.token_ids),
            "events": [_event_to_dict(e) for e in result.events],
        }
        with open(json_trace, "w") as f:
            json.dump(trace, f, indent=2)
        click.echo(f"Trace written to {json_trace}", err=True)


@main.command()
@click.option("--prompt", "-p", required=True, help="Input prompt text.")
@click.option("--max-tokens", default=256, show_default=True)
@click.option("--mock", is_flag=True, hidden=True, help="Use MockBackend (for testing).")
@click.option("--model", type=click.Path(exists=True), envvar="SPILLAGE_MODEL", default=None, help="Path to GGUF model file.")
@click.option("--n-gpu-layers", default=-1, show_default=True)
def greedy(prompt: str, max_tokens: int, mock: bool, model: str | None, n_gpu_layers: int) -> None:
    """Generate text using standard greedy (argmax) decoding."""
    if mock:
        backend = MockBackend(vocab_size=16)
    elif model:
        from .backend_native import NativeBackend
        backend = NativeBackend(model_path=model, n_gpu_layers=n_gpu_layers)
    else:
        click.echo("Error: provide --model /path/to/model.gguf or --mock.", err=True)
        raise SystemExit(1)

    result = generate_greedy(prompt, backend, max_tokens=max_tokens)
    click.echo(result.text)
    click.echo(
        f"\n--- {len(result.token_ids)} tokens, "
        f"{result.total_time_ms:.1f}ms total ---",
        err=True,
    )
