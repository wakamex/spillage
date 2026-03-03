import asyncio
import click
from spillage.backend import LlamaCppBackend
from spillage.sampler import MSSSampler
from spillage.scorer import MSSScoringConfig

SCENARIOS = [
    {
        "name": "Entity Swap",
        "prompt": "The founder of SpaceX is Elon Musk, but the founder of Blue Origin is ",
        "expected": ["Jeff Bezos", "Bezos"],
        "failure_trigger": "Elon Musk"
    },
    {
        "name": "Dead Person Bio",
        "prompt": "Is the actress Judi Dench alive? Yes, she is. Regarding her current status, she ",
        "expected": ["is still acting", "lives", "is alive", "active"],
        "failure_trigger": "died",
    },
    {
        "name": "Math Flow",
        "prompt": "What is 347 + 268? Let's calculate: 300+200=500, 40+60=100, 7+8=15. So 500+100+15 is ",
        "expected": ["615"],
        "failure_trigger": ["515", "715"]
    }
]

async def run_scenario(scenario, backend, k, beta, tau, threshold):
    print(f"\n=== Scenario: {scenario['name']} ===")
    
    # 1. Run Greedy Baseline (threshold=1.0 forces fast-path always)
    greedy_sampler = MSSSampler(backend, uncertainty_threshold=1.0)
    print("Greedy Output: ", end="", flush=True)
    greedy_text = ""
    async for token in greedy_sampler.generate(scenario['prompt'], max_tokens=30):
        print(token, end="", flush=True)
        greedy_text += token
    print()

    # 2. Run MSS Platinum
    config = MSSScoringConfig(beta=beta, tau=tau)
    mss_sampler = MSSSampler(backend, k=k, uncertainty_threshold=threshold, config=config)
    print("MSS Output:    ", end="", flush=True)
    mss_text = ""
    async for token in mss_sampler.generate(scenario['prompt'], max_tokens=30):
        print(token, end="", flush=True)
        mss_text += token
    print()

    # Comparison Logic
    if mss_text != greedy_text:
        print("⚡ DIVERGENCE DETECTED!")
    else:
        print("🤝 AGREEMENT (Both chose the same path)")

    # Simple validation for MSS
    passed = any(exp in mss_text for exp in scenario['expected'])
    if passed:
        print("MSS Result: ✅ PASS")
    else:
        print("MSS Result: ❌ FAIL")

@click.command()
@click.option("--server", "-s", default="http://localhost:8080", help="Llama.cpp server URL.")
@click.option("--k", "-k", default=3, help="Number of candidates.")
@click.option("--beta", "-b", default=2.0, help="Penalty.")
@click.option("--tau", "-t", default=4.2, help="Stability Threshold.")
@click.option("--threshold", "-u", default=0.92, help="Uncertainty threshold.")
def main(server, k, beta, tau, threshold):
    """Run Stress Tests for Min-Spill Search."""
    async def run_all():
        backend = LlamaCppBackend(server)
        for scenario in SCENARIOS:
            await run_scenario(scenario, backend, k, beta, tau, threshold)

    asyncio.run(run_all())

if __name__ == "__main__":
    main()
