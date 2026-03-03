import httpx
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Protocol, List, Optional

@dataclass
class LogitResult:
    token_ids: np.ndarray    # (V,) or (top_k,)
    logits: np.ndarray       # Raw logit values
    log_z: float             # logsumexp(logits)

class Backend(Protocol):
    async def get_logits(self, prompt: str, top_k: Optional[int] = None) -> LogitResult: ...
    async def get_logits_batch(self, prompts: List[str], top_k: Optional[int] = None) -> List[LogitResult]: ...

def logsumexp(x: np.ndarray) -> float:
    """Stable logsumexp implementation."""
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

class LlamaCppBackend:
    """
    Native llama.cpp backend interface.
    Prioritizes raw logits for exact log Z calculation.
    """
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.semaphore = asyncio.Semaphore(1) # Limit to 1 concurrent request for stability

    async def get_logits(self, prompt: str, top_k: Optional[int] = None) -> LogitResult:
        """Fetch logits/probs for a single prompt using the native /completion endpoint."""
        payload = {
            "prompt": prompt,
            "n_predict": 0,
            "n_probs": top_k if top_k else 100
        }
        
        async with self.semaphore:
            response = await self.client.post(f"{self.server_url}/completion", json=payload)
            response.raise_for_status()
            data = response.json()
        
        # Handle different server response formats (probs vs completion_probabilities)
        probs_data = data.get("probs") or (data.get("completion_probabilities", [{}])[0].get("top_logprobs"))
        # If n_probs was successful, we should have a 'probs' field
        if probs_data:
            top_logits = np.array([p.get("logit") or p.get("logprob", 0.0) for p in probs_data])
            log_z = logsumexp(top_logits)
            token_ids = np.array([p["id"] for p in probs_data])
            return LogitResult(token_ids=token_ids, logits=top_logits, log_z=log_z)

        else:
            raise ValueError(f"Server response missing probability fields: {data.keys()}")

    async def get_logits_batch(self, prompts: List[str], top_k: Optional[int] = None) -> List[LogitResult]:
        """Perform batched lookahead for k candidates."""
        import asyncio
        tasks = [self.get_logits(p, top_k) for p in prompts]
        return await asyncio.gather(*tasks)

    async def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text using the server's /detokenize endpoint."""
        payload = {"tokens": token_ids}
        response = await self.client.post(f"{self.server_url}/detokenize", json=payload)
        response.raise_for_status()
        return response.json().get("content", "")
