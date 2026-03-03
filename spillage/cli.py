import asyncio
import click
from .backend import LlamaCppBackend
from .sampler import MSSSampler
from .scorer import MSSScoringConfig

@click.command()
@click.option("--prompt", "-p", required=True, help="Initial prompt for generation.")
@click.option("--server", "-s", default="http://localhost:8080", help="Llama.cpp server URL.")
@click.option("--k", "-k", default=3, help="Number of lookahead candidates.")
@click.option("--beta", "-b", default=2.0, help="Consistency weight / Penalization strength.")
@click.option("--tau", "-t", default=4.2, help="Stability threshold (4.8 for 35B, 4.2 for 9B).")
@click.option("--threshold", "-u", default=0.92, help="Uncertainty threshold for Adaptive Gating.")
@click.option("--max-tokens", "-m", default=128, help="Maximum number of tokens to generate.")
def main(prompt, server, k, beta, tau, threshold, max_tokens):
    """Min-Spill Search (MSS) CLI."""
    
    async def run():
        backend = LlamaCppBackend(server)
        config = MSSScoringConfig(beta=beta, tau=tau)
        sampler = MSSSampler(backend, k=k, uncertainty_threshold=threshold, config=config)
        
        print(f"Prompt: {prompt}\n")
        print("Output: ", end="", flush=True)
        
        async for token in sampler.generate(prompt, max_tokens=max_tokens):
            print(token, end="", flush=True)
        print("\n")

    asyncio.run(run())

if __name__ == "__main__":
    main()
