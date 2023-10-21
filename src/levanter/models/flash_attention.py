# cf https://github.com/lucidrains/flash-attention-jax
# cf https://tridao.me/publications/flash2/flash2.pdf
# cf https://arxiv.org/pdf/2205.14135.pdf
from typing import Optional, Tuple

import equinox
import jax
import jax.numpy as jnp
from equinox import filter_eval_shape
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax.jax_utils import named_call

from levanter.models.attention import AttnMask


# TODO: tune
BLOCK_SIZE = 128


@named_call
def flash_attention(
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: Optional[AttnMask] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: int = BLOCK_SIZE,
):
    """
    Flash Attention impl, vaguely following the v2 paper.

    Args:
        Key: axis of key dim.
    """
    if not inference and dropout > 0 and key is None:
        raise ValueError("key must be provided for training")

    if dropout < 0 or dropout > 1:
        raise ValueError(f"invalid dropout {dropout}")

    # premultiply by 1/sqrt(d_k) for normal dot product attention
    q = q * jax.lax.rsqrt(float(q.axis_size(Key)))

    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)

    return _flash_attention(
        (q, k, v), QPos, KPos, Key, mask, dropout, inference=inference, key=key, block_size=block_size
    )


@equinox.filter_custom_vjp
def _flash_attention(
    qkv: Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray],
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    mask: Optional[AttnMask] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: int,
) -> hax.NamedArray:
    return _flash_attention_forward(
        None, qkv, QPos, KPos, Key, mask, dropout, inference=inference, key=key, block_size=block_size
    )[0]


@named_call
def _flash_attention_forward(
    ignore,
    qkv,
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.AxisSelector,
    mask: Optional[AttnMask],
    dropout: float,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray],
    block_size: int,
):
    del ignore
    q, k, v = qkv
    if QPos.size % block_size != 0:
        raise ValueError(f"q axis size {q.axis_size(QPos)} is not a multiple of {block_size}")
    if KPos.size % block_size != 0:
        raise ValueError(f"k axis size {k.axis_size(KPos)} is not a multiple of {block_size}")

    QPosBlock = QPos.resize(block_size)  # Br in the paper
    KPosBlock = KPos.resize(block_size)  # Bc in the paper

    # number of blocks for Q and K
    Tr = QPos.size // block_size
    Tc = KPos.size // block_size

    q_batch_axes: Tuple[hax.Axis, ...] = hax.eliminate_axes(q.axes, (QPos, Key))

    # output variables: O is the attention output, ell is the per-position log normalizer
    o_shape = _infer_attention_output_block_shape(QPos, KPos, Key, q, k, v)
    o = hax.zeros(o_shape, q.dtype)
    o = hax.auto_sharded(o)
    ell = hax.zeros((*q_batch_axes, QPos))
    ell = hax.auto_sharded(ell)

    @named_call
    def do_o_block(state):
        i, o, ell = state

        # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
        q_i = q.slice(QPos, QPosBlock, i * block_size)

        # Step 2: init O_i = 0, sumexp_i = 0, max_i = -inf
        o_i = o.slice(QPos, QPosBlock, i * block_size)
        sumexp_i = hax.zeros(q_batch_axes + (QPosBlock,), q.dtype)
        max_i = hax.full(q_batch_axes + (QPosBlock,), -jnp.inf, q.dtype)

        @named_call
        def do_qk_block(state):
            """computes softmax(Q_i K_j^T) V_j"""
            # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
            #         K and V into 𝑇𝑐 = \ceil(𝑁/Bc) blocks of size Bc x d each.
            i, j, o_i, q_i, sumexp_i, old_max_i = state
            k_j = k.slice(KPos, KPosBlock, j * block_size)
            v_j = v.slice(KPos, KPosBlock, j * block_size)

            # TODO: precision
            # Step 8: compute Sij = QiKj^T
            attn_ij = hax.dot(Key, q_i, k_j)

            if mask is not None:
                mask_ij = _materialize_mask_slice(mask, i, j, QPos, KPos, QPosBlock, KPosBlock, block_size)
                attn_ij = hax.where(mask_ij, attn_ij, -1e10)

            # TODO: block causal

            if dropout > 0 and not inference:
                attn_ij = hax.nn.dropout(attn_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            # Step 9: Compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j)), P_i^j = exp(S_i^j - m_i^j),
            # ...    l_i^j = exp(m_i^{j-1} - m_i^j) + rowsum(P_i^j)
            max_i = hax.maximum(old_max_i, hax.max(attn_ij, axis=KPosBlock))
            P_ij = hax.exp(attn_ij - max_i)

            exp_diff = hax.exp(old_max_i - max_i)
            sumexp_i = exp_diff * sumexp_i + hax.sum(P_ij, axis=KPosBlock)

            # Step 10: Compute O_i = diag(exp(m_i^{j-1} - m_i^j) O_i + P_i^j V_j
            o_i = exp_diff * o_i + hax.dot(KPosBlock, P_ij, v_j)

            return (i, j + 1, o_i, q_i, sumexp_i, max_i)

        _, _, o_i, _, sumexp_i, max_i = jax.lax.while_loop(
            lambda state: state[1] < Tc, do_qk_block, (i, 0, o_i, q_i, sumexp_i, max_i)
        )

        # Step 12: compute O_i = diag(\ell_i^{Tc})^{-1} O_i^{Tc}
        o_i = o_i / sumexp_i
        # Step 13: compute L_i = m_i^{Tc} + log(\ell_i^{Tc})
        ell_i = max_i + hax.log(sumexp_i)

        o = o.updated_slice({QPos: i * block_size}, o_i)
        ell = ell.updated_slice({QPos: i * block_size}, ell_i.astype(ell.dtype))

        return i + 1, o, ell

    # o, ell = hax.map(do_o_block, Tr)(jnp.arange(Tr.size))
    _, o, ell = jax.lax.while_loop(lambda state: state[0] < Tr, do_o_block, (0, o, ell))

    return o, (o, ell)


@named_call
def _flash_attention_backward(
    residuals,
    grad_in: hax.NamedArray,
    ignore,
    qkv,
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: int,
):
    del ignore
    O, L = residuals
    q, k, v = qkv
    dO = grad_in

    Tr = QPos.size // block_size
    Tc = KPos.size // block_size

    if isinstance(mask, hax.NamedArray):
        mask = mask.broadcast_axis((QPos, KPos))  # make sure mask is broadcastable

    KPosBlock = KPos.resize(block_size)
    QPosBlock = QPos.resize(block_size)

    # Compute D = rowsum(dO * O), write D to HBM and divide it into Tr blocks of size Br each.
    # in the FA2 paper D is said to be \in R^{d}, but that doesn't make sense.
    # Triton impl has it as R^{QPos}, which makes more sense.
    D = hax.sum(dO * O, axis=Key)

    dQ = (q * 0.0).astype(q.dtype)
    dK = (k * 0.0).astype(k.dtype)
    dV = (v * 0.0).astype(v.dtype)

    @named_call
    def do_kv_block(state):
        j, dQ, dK, dV = state
        k_j = k.slice(KPos, KPosBlock, j * block_size)
        v_j = v.slice(KPos, KPosBlock, j * block_size)

        dK_j = dK.slice(KPos, KPosBlock, j * block_size)
        dV_j = dV.slice(KPos, KPosBlock, j * block_size)

        @named_call
        def do_inner_block(state):
            i, j, dQ, dK_j, dV_j = state
            q_i = q.slice(QPos, QPosBlock, i * block_size)

            dQ_i = dQ.slice(QPos, QPosBlock, i * block_size)
            dO_i = dO.slice(QPos, QPosBlock, i * block_size)
            L_i = L.slice(QPos, QPosBlock, i * block_size)
            D_i = D.slice(QPos, QPosBlock, i * block_size)

            # TODO: precision
            attn_ij = hax.dot(Key, q_i, k_j)

            if dropout > 0 and not inference:
                attn_ij = hax.nn.dropout(attn_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            if mask is not None:
                mask_ij = _materialize_mask_slice(mask, i, j, QPos, KPos, QPosBlock, KPosBlock, block_size)
                attn_ij = hax.where(mask_ij, attn_ij, -1e10)

            p_ij = hax.exp(attn_ij - L_i)

            if dropout > 0 and not inference:
                p_ij = hax.nn.dropout(p_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            dP_ij = hax.dot(Key, dO_i, v_j)
            dAttn_ij = p_ij * (dP_ij - D_i)
            dAttn_ij = dAttn_ij.astype(dQ_i.dtype)

            dV_j = dV_j + hax.dot(QPosBlock, p_ij, dO_i).astype(dV_j.dtype)
            dK_j = dK_j + hax.dot(QPosBlock, dAttn_ij, q_i).astype(dK_j.dtype)

            dQ_i = dQ_i + hax.dot(KPosBlock, dAttn_ij, k_j).astype(dQ.dtype)
            # dQ[i*block_size:(i+1)*block_size] = dQi
            dQ = dQ.updated_slice({QPos: i * block_size}, dQ_i)

            return i + 1, j, dQ, dK_j, dV_j

        # dQ, dK_j, dV_j = hax.fold(do_inner_block, Tr)((dQ, dK_j, dV_j), jnp.arange(Tr.size))
        i, j, dQ, dK_j, dV_j = jax.lax.while_loop(lambda state: state[0] < Tr, do_inner_block, (0, j, dQ, dK_j, dV_j))

        dK = dK.updated_slice({KPos: j * block_size}, dK_j)
        dV = dV.updated_slice({KPos: j * block_size}, dV_j)

        return j + 1, dQ, dK, dV

    # dQ, (dK, dV) = hax.scan(do_kv_block, Tc)(dQ, jnp.arange(Tc.size))
    j, dQ, dK, dV = jax.lax.while_loop(lambda state: state[0] < Tc, do_kv_block, (0, dQ, dK, dV))
    return dQ.rearrange(q.axes), dK.rearrange(k.axes), dV.rearrange(v.axes)


_flash_attention.def_fwd(_flash_attention_forward)
_flash_attention.def_bwd(_flash_attention_backward)


def _infer_attention_output_block_shape(QPosBlock, KPos, Key, q_i, k, v):
    out_shape = filter_eval_shape(hnn.attention.dot_product_attention, QPosBlock, KPos, Key, q_i, k, v)
    return out_shape.axes


def _materialize_mask_slice(mask, i, j, QPos, KPos, QPosBlock, KPosBlock, block_size):
    if isinstance(mask, hax.NamedArray):
        mask_ij = mask.slice(QPos, QPosBlock, i * block_size).slice(KPos, KPosBlock, j * block_size)
    else:
        mask_ij = mask.slice(QPos, i * block_size, block_size).slice(KPos, j * block_size, block_size)
        mask_ij = mask_ij.materialize()

    return mask_ij
