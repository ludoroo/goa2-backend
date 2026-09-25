import subprocess
import sys


def test_classic_search_and_server_modules_are_learned_model_neutral() -> None:
    code = """
import automata.search
import goa2.server.bot_factory
import goa2.server.bounded_compute
import goa2.server.bots
import sys
blocked = (
    'automata.models', 'automata.observation', 'torch',
    'automata.evaluation.learned_value', 'automata.search.learned_policy',
    'goa2.server.neural_rollout',
)
loaded = [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in blocked)]
assert not loaded, loaded
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
