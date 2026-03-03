from mss.backend import MockBackend
from mss.decoder import MSSConfig, MinSpillDecoder


def test_decoder_uses_greedy_fast_path_when_confident() -> None:
    backend = MockBackend(
        {
            tuple(): [8.0, 1.0, -1.0],
            (0,): [-1.0, -1.0, 3.0],
        },
        eos_token_id=2,
    )
    decoder = MinSpillDecoder(backend, MSSConfig(max_tokens=1))
    emitted, decisions = decoder.generate([])
    assert emitted == [0]
    assert decisions[0].strategy == "greedy_fast_path"


def test_decoder_reranks_with_spill_penalty() -> None:
    backend = MockBackend(
        {
            tuple(): [2.30, 2.20, -2.0],
            (0,): [5.0, 5.0, 5.0],
            (1,): [1.0, 0.0, -1.0],
            (2,): [0.0, 0.0, 0.0],
        },
        eos_token_id=2,
    )
    config = MSSConfig(k=2, beta=2.0, tau=0.0, adaptive_k=False, max_tokens=1)
    decoder = MinSpillDecoder(backend, config)

    emitted, decisions = decoder.generate([])

    assert emitted == [1]
    assert decisions[0].strategy == "mss_topk"
    assert len(decisions[0].candidates) == 2


def test_generate_trace_reports_timing_and_steps() -> None:
    backend = MockBackend(
        {
            tuple(): [8.0, 1.0, -1.0],
            (0,): [-1.0, -1.0, 3.0],
        },
        eos_token_id=2,
    )
    decoder = MinSpillDecoder(backend, MSSConfig(max_tokens=2))
    trace = decoder.generate_trace([])
    assert trace.emitted == (0, 2)
    assert len(trace.decisions) == 2
    assert len(trace.step_durations_s) == 2
    assert trace.ttft_s >= 0.0
    assert trace.total_duration_s >= trace.ttft_s


def test_decoder_uses_next_steps_for_candidate_evaluation() -> None:
    class CountingBackend(MockBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_calls = 0

        def next_steps(self, contexts, *, top_n=None, require_logits=False):
            self.batch_calls += 1
            return super().next_steps(contexts, top_n=top_n, require_logits=require_logits)

    backend = CountingBackend(
        {
            tuple(): [2.3, 2.2, -2.0],
            (0,): [5.0, 5.0, 5.0],
            (1,): [1.0, 0.0, -1.0],
            (2,): [0.0, 0.0, 0.0],
        },
        eos_token_id=2,
    )
    config = MSSConfig(k=2, beta=2.0, tau=0.0, adaptive_k=False, max_tokens=1)
    decoder = MinSpillDecoder(backend, config)

    emitted, _ = decoder.generate([])
    assert emitted == [1]
    assert backend.batch_calls == 1
