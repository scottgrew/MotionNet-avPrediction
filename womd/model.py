"""The network. Everything a predicted agent sees arrives in its own frame
(womd/views.py): each agent's history and each map chunk become one token, a
transformer encoder relates the tokens, and 54 anchor queries decode one
future each through a transformer decoder.
"""

from __future__ import annotations

from typing import List, NamedTuple

import torch
from torch import Tensor, nn

from womd import layout
from womd.views import TargetInputs

# Network size, chosen values. Tokens, queries and attention layers
# share one hidden width.
HIDDEN_DIM = 512
FEEDFORWARD_DIM = 4 * HIDDEN_DIM
ATTENTION_HEAD_COUNT = 4
SCENE_ATTENTION_ROUNDS = 6
DECODER_ROUNDS = 6

# The decoder predicts one future per anchor, 54 per object type;
# womd/pruning.py later keeps the 6 that Waymo scores.
QUERY_COUNT = 54
# A chunk's dots are read relative to its centre in units of half a
# chunk's length, so a 20 m curve spans about -1 to 1.
CHUNK_HALF_LENGTH_METRES = (
    layout.MAP_CHUNK_DOTS * layout.MAP_POINT_SPACING_METRES / 2
)
# Predicted log sigma is clamped to 0.2 m .. 148 m; without a
# floor the likelihood rewards ever-narrower Gaussians.
MINIMUM_LOG_STANDARD_DEVIATION = -1.609
MAXIMUM_LOG_STANDARD_DEVIATION = 5.0

# torch's own pre-norm transformer layers, with no dropout.
TRANSFORMER_LAYER_SETTINGS = {
    "d_model": HIDDEN_DIM,
    "nhead": ATTENTION_HEAD_COUNT,
    "dim_feedforward": FEEDFORWARD_DIM,
    "dropout": 0.0,
    "activation": "relu",
    "batch_first": True,
    "norm_first": True,
}


class ModePredictions(NamedTuple):
    """What the decoder returns per mode, a path and its per-step log sigma in
    metres, a confidence logit and the anchor it grew from.
    """

    trajectories: Tensor  # (batch, 54, 80, 2)
    log_standard_deviation: Tensor  # (batch, 54, 80, 2)
    confidence_logits: Tensor  # (batch, 54)
    anchors: Tensor  # (batch, 54, 2)


RoundPredictions = List[ModePredictions]


class Memory(NamedTuple):
    """The encoded tokens of each target's scene, agents (the target first)
    then map chunks, and a mask of which tokens are real.
    """

    tokens: Tensor  # (targets, 48 + 256, hidden)
    present: Tensor  # (targets, 48 + 256)


def agent_feature_divisors() -> Tensor:
    """Per-feature scale so position (metres), velocity (m/s) and dimensions
    (metres) are normalised before they enter the network.
    """
    divisors = torch.ones(layout.AGENT_FEATURE_DIM)
    divisors[layout.AGENT_POSITION] = layout.DISTANCE_NORMALISER_METRES
    divisors[layout.AGENT_VELOCITY] = (
        layout.VELOCITY_NORMALISER_METRES_PER_SECOND
    )
    divisors[layout.AGENT_DIMENSIONS] = layout.DIMENSION_NORMALISER_METRES
    return divisors


def map_feature_divisors() -> Tensor:
    """Per-feature scale for map rows: position relative to its chunk's
    centre, stop point in the target's frame in metres, and speed limit in
    miles per hour.
    """
    divisors = torch.ones(layout.MAP_FEATURE_DIM)
    divisors[layout.MAP_POSITION] = CHUNK_HALF_LENGTH_METRES
    divisors[layout.MAP_SPEED_LIMIT] = (
        layout.SPEED_LIMIT_NORMALISER_MILES_PER_HOUR
    )
    divisors[layout.MAP_STOP_POINT] = layout.DISTANCE_NORMALISER_METRES
    return divisors


def two_layer_mlp(input_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    )


def predicted_type_index(history: Tensor) -> Tensor:
    """Each agent's predicted type index, from the one-hot object type at the
    current history step.
    """
    type_one_hot = history[:, layout.CURRENT_STEP_INDEX, layout.AGENT_TYPE]
    return type_one_hot[:, : layout.NUM_OBJECT_TYPES].argmax(dim=-1)


def agent_reachable_distance(history: Tensor) -> Tensor:
    """Upper bound on how far an agent could travel over the future horizon at
    constant maximum acceleration from its current speed.
    """
    speed = history[:, layout.CURRENT_STEP_INDEX, layout.AGENT_VELOCITY].norm(
        dim=-1
    )
    horizon = layout.FUTURE_HORIZON_SECONDS
    acceleration = layout.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED
    # d = v T + a T^2 / 2 for speed v, horizon T and acceleration a.
    return speed * horizon + 0.5 * acceleration * horizon**2


def anchor_lines(anchors: Tensor) -> Tensor:
    """Each anchor endpoint (..., 2) as a constant-speed straight line from
    the agent, (..., 80, 2): future step k sits at k / 80 of the endpoint.
    The decoder adds its offsets to these lines and the loss assigns each
    logged future to the nearest one, so both must use this one rule.
    """
    steps = torch.arange(1, layout.FUTURE_STEPS + 1, device=anchors.device)
    ramp = steps.to(anchors.dtype) / layout.FUTURE_STEPS
    return anchors[..., None, :] * ramp[:, None]


class ModeDecoder(nn.Module):
    """Decodes the 54 anchor-conditioned trajectory modes and their confidences
    from the memory MotionPredictor.encode builds. Each query is an embedding
    of its anchor endpoint alone, so mode k always belongs to anchor k. The same
    heads read the queries after every round, so training can ask every
    round for a usable answer.
    """

    def __init__(self, unit_anchors: Tensor) -> None:
        super().__init__()
        expected_shape = (layout.NUM_OBJECT_TYPES, QUERY_COUNT, 2)
        if unit_anchors.shape != expected_shape:
            raise ValueError(
                f"unit_anchors has shape {tuple(unit_anchors.shape)},"
                f" not {expected_shape}"
            )
        # Anchor endpoints in metres, despite the name, which saved
        # checkpoints and anchor files depend on.
        self.register_buffer(
            "unit_anchors", unit_anchors.detach().clone().float()
        )
        self.anchor_query = two_layer_mlp(2)
        self.scene_norm = nn.LayerNorm(HIDDEN_DIM)
        self.rounds = nn.ModuleList(
            nn.TransformerDecoderLayer(**TRANSFORMER_LAYER_SETTINGS)
            for _ in range(DECODER_ROUNDS)
        )
        self.head_norm = nn.LayerNorm(HIDDEN_DIM)
        # Per future step, an x, y offset and an x, y log sigma.
        self.trajectory_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 2 * layout.FUTURE_STEPS * 2),
        )
        # There is no final bias because one bias shared by all modes
        # cancels in the softmax over modes and could never be trained.
        self.confidence_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 1, bias=False),
        )

    def decode(self, queries: Tensor, anchors: Tensor, lines: Tensor):
        """Trajectories and log sigmas (batch, modes, steps, 2) in metres and
        confidence logits (batch, modes) from one round's queries. lines is
        anchor_lines(anchors), built once per forward pass.
        """
        queries = self.head_norm(queries)
        # Cast up so the cumulative sum below runs in float32 even under
        # mixed precision.
        head_shape = queries.shape[:2] + (2, layout.FUTURE_STEPS, 2)
        head_output = self.trajectory_head(queries).float().view(head_shape)
        step_displacements, log_sigma = head_output.unbind(dim=2)
        # A mode is its anchor's straight constant-speed line plus the running
        # sum of the predicted per-step offsets.
        trajectories = step_displacements.cumsum(dim=-2) + lines
        log_sigma = log_sigma.clamp(
            MINIMUM_LOG_STANDARD_DEVIATION, MAXIMUM_LOG_STANDARD_DEVIATION
        )
        confidence_logits = self.confidence_head(queries).squeeze(-1)
        return ModePredictions(
            trajectories, log_sigma, confidence_logits, anchors
        )

    def forward(self, memory: Memory, type_index: Tensor) -> RoundPredictions:
        """Each round's modes, first round first; the last is the answer."""
        anchors = self.unit_anchors[type_index]
        lines = anchor_lines(anchors)
        scaled_anchors = anchors / layout.DISTANCE_NORMALISER_METRES
        queries = self.anchor_query(scaled_anchors)
        normed_tokens = self.scene_norm(memory.tokens)
        padding = ~memory.present
        per_round = []
        for decoder_round in self.rounds:
            queries = decoder_round(
                queries, normed_tokens, memory_key_padding_mask=padding
            )
            per_round.append(self.decode(queries, anchors, lines))
        return per_round


class MotionPredictor(nn.Module):
    """The scene encoder and the mode decoder. The encoder's weights live
    here: an MLP over each agent's 11 history steps, an MLP per map dot that
    is max-pooled into chunk tokens, and the transformer encoder that relates
    all the tokens. Everything arrives already in the target's frame.
    """

    def __init__(self, unit_anchors: Tensor) -> None:
        super().__init__()
        self.register_buffer("agent_divisors", agent_feature_divisors())
        self.register_buffer("map_divisors", map_feature_divisors())
        crossing_codes = layout.NUM_BOUNDARY_CROSSING_CODES
        self.register_buffer("crossing_code_rows", torch.eye(crossing_codes))
        step_width = layout.AGENT_FEATURE_DIM + 1
        self.agent_network = two_layer_mlp(layout.HISTORY_STEPS * step_width)
        plain_columns = layout.MAP_LEFT_BOUNDARY_CROSSING
        self.dot_network = two_layer_mlp(plain_columns + 2 * crossing_codes)
        self.chunk_pose_projection = nn.Linear(4, HIDDEN_DIM)
        signal_width = layout.POLYLINE_SIGNAL_DIM
        self.signal_projection = nn.Linear(signal_width, HIDDEN_DIM, bias=False)
        layer = nn.TransformerEncoderLayer(**TRANSFORMER_LAYER_SETTINGS)
        self.encoder = nn.TransformerEncoder(
            layer, SCENE_ATTENTION_ROUNDS, enable_nested_tensor=False
        )
        self.mode_decoder = ModeDecoder(unit_anchors)

    @property
    def unit_anchors(self) -> Tensor:
        return self.mode_decoder.unit_anchors

    def agent_tokens(self, inputs: TargetInputs) -> Tensor:
        """One token per agent slot (B, 48, hidden) from its history steps and
        one validity flag per step. Invalid steps are replaced by zeros
        rather than multiplied by zero, so a NaN or inf stored there cannot
        leak through.
        """
        history = inputs.agent_history
        step_is_valid = inputs.agent_history_mask.unsqueeze(-1)
        zeros = torch.zeros_like(history)
        observed = torch.where(step_is_valid, history, zeros)
        scaled = observed / self.agent_divisors
        validity = step_is_valid.to(history.dtype)
        with_validity = torch.cat([scaled, validity], dim=-1)
        return self.agent_network(with_validity.flatten(start_dim=-2))

    def dot_tokens(self, map_rows: Tensor) -> Tensor:
        """One token per map dot. The two crossing codes are category numbers,
        not quantities, so each is widened to a one-hot first.
        """
        crossing = layout.MAP_LEFT_BOUNDARY_CROSSING
        scaled = map_rows / self.map_divisors
        plain = scaled[..., :crossing]
        crossing_codes = map_rows[..., crossing:].long()
        one_hot = self.crossing_code_rows[crossing_codes].flatten(start_dim=-2)
        features = torch.cat([plain, one_hot.to(scaled.dtype)], dim=-1)
        return self.dot_network(features)

    def chunk_tokens(self, inputs: TargetInputs) -> tuple[Tensor, Tensor]:
        """Chunk tokens (B, 256, hidden) and whether each chunk has any dot.
        A token is its dots relative to the chunk's centre, max-pooled, plus
        the centre, the mean direction and the chunk's light history. Masked
        dots and absent chunks are replaced by zeros first, so a NaN stored
        under a mask cannot reach an index or a token.
        """
        dot_mask = inputs.chunk_dot_mask
        stored_rows = inputs.chunk_rows
        rows_zeros = torch.zeros_like(stored_rows)
        rows = torch.where(dot_mask.unsqueeze(-1), stored_rows, rows_zeros)
        present = dot_mask.any(dim=-1)
        stored_signals = inputs.chunk_signal_history
        signal_zeros = torch.zeros_like(stored_signals)
        chunk_is_present = present[..., None, None]
        signal_history = torch.where(
            chunk_is_present, stored_signals, signal_zeros
        )
        is_dot = dot_mask.unsqueeze(-1).to(rows.dtype)
        dot_count = is_dot.sum(dim=-2).clamp_min(1.0)
        positions = rows[..., layout.MAP_POSITION]
        directions = rows[..., layout.MAP_DIRECTION]
        centre = (positions * is_dot).sum(dim=-2) / dot_count
        mean_direction = (directions * is_dot).sum(dim=-2) / dot_count
        local = positions - centre.unsqueeze(-2)
        local_rows = torch.cat([local, rows[..., 2:]], dim=-1)

        dots = self.dot_tokens(local_rows)
        dots = dots.masked_fill(~dot_mask.unsqueeze(-1), float("-inf"))
        pooled = dots.max(dim=-2).values
        pooled = pooled.masked_fill(~present.unsqueeze(-1), 0.0)
        scaled_centre = centre / layout.DISTANCE_NORMALISER_METRES
        pose = torch.cat([scaled_centre, mean_direction], dim=-1)
        signals = signal_history.flatten(start_dim=-2)
        tokens = pooled + self.chunk_pose_projection(pose)
        return tokens + self.signal_projection(signals), present

    def encode(self, inputs: TargetInputs) -> Memory:
        """The agent and chunk tokens after self-attention, and which of them
        are real.
        """
        agent_tokens = self.agent_tokens(inputs)
        agent_present = inputs.agent_history_mask.any(dim=-1)
        chunk_tokens, chunk_present = self.chunk_tokens(inputs)
        tokens = torch.cat([agent_tokens, chunk_tokens], dim=1)
        present = torch.cat([agent_present, chunk_present], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~present)
        return Memory(encoded, present)

    def forward(self, inputs: TargetInputs) -> RoundPredictions:
        """One ModePredictions per decoder round, with anchors chosen by each
        target's predicted type; the last round is the model's answer.
        Training reads every round, so this is the one call DDP wraps.
        """
        memory = self.encode(inputs)
        own_history = inputs.agent_history[:, 0]
        type_index = predicted_type_index(own_history)
        return self.mode_decoder(memory, type_index)


# ------------------------------------------------------------------
# DATA FLOW for B predicted agents, hidden width 512, all in each
# agent's own frame (womd/views.py)
#
#   48 agent histories --MLP over 11 steps-->     48 tokens  \
#                                                              > (B, 304, 512)
#   256 map chunks --MLP per dot, max-pool,                  /
#                    + centre, direction, lights--> 256 tokens
#                          |
#            6 x pre-norm self-attention layer
#                          |
#   54 queries, each an MLP of its anchor endpoint
#                          |
#            6 x [ self-attention over the 54 modes,
#                  cross-attention to the tokens,
#                  feedforward ]
#                          |
#            +-------------+--------------+
#            |                            |
#     trajectory head              confidence head
#     offsets and log sigma,       logits (B, 54)
#     (B, 54, 80, 2) each
#            |
#     path = the anchor's straight line + running sum of offsets
# ------------------------------------------------------------------
