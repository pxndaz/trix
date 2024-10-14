from typing import NamedTuple
import jax
import jax.numpy as jnp

class AttnStats(NamedTuple):
    entropy: jax.Array  # (bsz, n_layers, num_heads)
    varentropy: jax.Array  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int
    prev_varentropy: jax.Array  # Buffer for last 3 steps of varentropy

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
            varentropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
            prev_varentropy=jnp.zeros((bsz, n_layers, 3), dtype=jnp.float32),  # Rolling buffer for 3 steps
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads

    def update(self, scores: jax.Array, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = jax.nn.softmax(scores, axis=-1)
        new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
        new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)

        # Shift the buffer (keep last 3 steps)
        updated_prev_varentropy = jnp.roll(self.prev_varentropy[:, layer_idx, :], shift=-1, axis=-1)
        updated_prev_varentropy = updated_prev_varentropy.at[:, -1].set(new_varentropy)

        # Detect 3 consecutive jumps
        varentropy_jumps = jnp.diff(updated_prev_varentropy, axis=-1)
        if (varentropy_jumps > some_threshold).all():  # Replace 'some_threshold' with the desired threshold
            trigger_feedback()  # Implement feedback trigger logic

        updated_stats = self._replace(
            entropy=self.entropy.at[:, layer_idx, :].set(new_entropy),
            varentropy=self.varentropy.at[:, layer_idx, :].set(new_varentropy),
            prev_varentropy=self.prev_varentropy.at[:, layer_idx, :].set(updated_prev_varentropy)
        )
        return updated_stats
