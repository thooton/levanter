import dataclasses
from dataclasses import dataclass
import equinox as eqx
import jax
import haliax as hax
import haliax.nn as hnn
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.compat.torch_serialization import StateDictSerializationMixin
from levanter.models.lm_model import LmConfig, LmHeadModel, LmExample
from typing import Optional

@LmConfig.register_subclass("aether")
@dataclass(frozen=True)
class AetherConfig(LmConfig):
    model_dim: int
    ff_dim: int
    query_count: int
    kv_count: int
    layer_count: int
    norm_eps: float
    seq_len: int
    vocab_size: int
    gradient_checkpointing: bool

    seq_axis = property(lambda self: hax.Axis(name="seq", size=self.seq_len))
    kv_seq_axis = property(lambda self: self.seq_axis.alias("kv_seq"))
    model_axis = property(lambda self: hax.Axis(name="model", size=self.model_dim))
    embed_axis = property(lambda self: self.model_axis.alias("embed"))
    ff_axis = property(lambda self: hax.Axis(name="ff", size=self.ff_dim))
    query_axis = property(lambda self: hax.Axis(name="query", size=self.query_count))
    kv_axis = property(lambda self: hax.Axis(name="kv", size=self.kv_count))
    kv_repeat_axis = property(lambda self: hax.Axis(name="kv_repeat", size=self.query_count // self.kv_count))
    layer_axis = property(lambda self: hax.Axis(name="layer", size=self.layer_count))
    head_axis = property(lambda self: hax.Axis(name="head", size=self.model_dim // self.query_count))
    vocab_axis = property(lambda self: hax.Axis(name="vocab", size=self.vocab_size))
    
    @property
    def model_type(cls):
        return Aether
    @property
    def KeyPos(self):
        return self.kv_seq_axis
    @property
    def Pos(self):
        return self.seq_axis
    def build(self, Vocab, *, key):
        return Aether.init(self, key)

def precompute_rope(head_axis, seq_axis):
    with jax.ensure_compile_time_eval():
        assert (head_axis.size % 2) == 0
        head_half_axis = head_axis.resize(head_axis.size // 2)
        inv_freq = 1.0 / (10000.0 ** (hax.arange(head_half_axis, step=2) / head_axis.size))
        freqs = hax.arange(seq_axis) * inv_freq.broadcast_axis(seq_axis)
        emb = hax.concatenate(head_axis, (freqs, freqs))
        sin = hax.sin(emb)
        cos = hax.cos(emb)
    return jax.lax.stop_gradient(sin), jax.lax.stop_gradient(cos)

def rotate_half(x):
    axis = x.axes[-1]
    x1 = x[axis, :axis.size//2]
    x2 = x[axis, axis.size//2:]
    x = hax.concatenate(axis, (-x2, x1))
    return x

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

def precompute_mask(seq_axis, kv_seq_axis):
    with jax.ensure_compile_time_eval():
        mask = hax.full((seq_axis, kv_seq_axis), -1e9)
        mask = hax.triu(mask, seq_axis, kv_seq_axis, 1)
    return jax.lax.stop_gradient(mask)

class FFN(eqx.Module, StateDictSerializationMixin):
    wg: hnn.Linear
    wu: hnn.Linear
    wd: hnn.Linear
    @staticmethod
    def init(model_axis, ff_axis, key):
        kg, ku, kd = jax.random.split(key, 3)
        wg = hnn.Linear.init(In=model_axis, Out=ff_axis, key=kg, use_bias=False)
        wu = hnn.Linear.init(In=model_axis, Out=ff_axis, key=ku, use_bias=False)
        wd = hnn.Linear.init(In=ff_axis, Out=model_axis, key=kd, use_bias=False)
        return FFN(wg, wu, wd)
    @named_call
    def __call__(self, x):
        x = hnn.silu(self.wg(x)) * self.wu(x)
        x = self.wd(x)
        return x

class Attention(eqx.Module, StateDictSerializationMixin):
    conf: AetherConfig = eqx.static_field()
    wq: hnn.Linear
    wk: hnn.Linear
    wv: hnn.Linear
    wo: hnn.Linear
    @staticmethod
    def init(conf, key):
        kq, kk, kv, ko = jax.random.split(key, 4)
        wq = hnn.Linear.init(In=conf.model_axis, Out=(conf.kv_repeat_axis, conf.kv_axis, conf.head_axis), key=kq, use_bias=False)
        wk = hnn.Linear.init(In=conf.model_axis, Out=(conf.kv_axis, conf.head_axis), key=kk, use_bias=False)
        wv = hnn.Linear.init(In=conf.model_axis, Out=(conf.kv_axis, conf.head_axis), key=kv, use_bias=False)
        wo = hnn.Linear.init(In=(conf.kv_repeat_axis, conf.kv_axis, conf.head_axis), Out=conf.model_axis, key=ko, use_bias=False)
        return Attention(conf, wq, wk, wv, wo)
    @named_call
    def __call__(self, x, mask, sin, cos):
        conf = self.conf
        batch_axis = x.axes[0]
        # (batch_size, seq_len, kv_repeat, kv_count, head_dim)
        q = self.wq(x)
        # (batch_size, kv_repeat, kv_count, seq_len, head_dim)
        q = q.rearrange((batch_axis, conf.kv_repeat_axis, conf.kv_axis, conf.seq_axis, conf.head_axis))
        # (batch_size, seq_len, kv_count, head_dim)
        k, v = self.wk(x), self.wv(x)
        # (batch_size, kv_count, seq_len, head_dim)
        k, v = (
            t.rearrange((batch_axis, conf.kv_axis, conf.seq_axis, conf.head_axis))
            for t in (k, v)
        )
        q, k = (
            apply_rope(t, sin, cos)
            for t in (q, k)
        )
        k = k * (conf.head_axis.size ** -0.5)
        # (batch_size, kv_count, kv_seq_len, head_dim)
        k, v = (
            t.rename({conf.seq_axis: conf.kv_seq_axis})
            for t in (k, v)
        )
        # (batch_size, kv_repeat, kv_count, seq_len, kv_seq_len)
        a = hax.dot(conf.head_axis, q, k)
        a = a + mask
        a = hnn.softmax(a, axis=conf.kv_seq_axis)
        # (batch_size, kv_repeat, kv_count, seq_len, head_dim)
        y = hax.dot(conf.kv_seq_axis, a, v)
        # (batch_size, seq_len, kv_repeat, kv_count, head_dim)
        y = y.rearrange((batch_axis, conf.seq_axis, conf.kv_repeat_axis, conf.kv_axis, conf.head_axis))
        # (batch_size, seq_len, model_dim)
        y = self.wo(y)
        return y

class RMSNorm(eqx.Module, StateDictSerializationMixin):
    axis: hax.Axis = eqx.static_field()
    eps: float = eqx.static_field()
    scale: hax.NamedArray
    @staticmethod
    def init(axis, eps):
        scale = hax.ones(axis)
        return RMSNorm(axis, eps, scale)
    @named_call
    def __call__(self, x):
        dtype = x.dtype
        x = x.astype("float32")
        x = x * hax.rsqrt(
            hax.mean(hax.square(x), axis=self.axis)
            + self.eps
        )
        x = x.astype(dtype)
        x = x * self.scale
        return x

class Block(eqx.Module, StateDictSerializationMixin):
    ln_1: RMSNorm
    attn: Attention
    ln_2: RMSNorm
    ffn: FFN
    @staticmethod
    def init(conf, key):
        ka, kf = jax.random.split(key, 2)
        ln_1 = RMSNorm.init(conf.model_axis, conf.norm_eps)
        attn = Attention.init(conf, ka)
        ln_2 = RMSNorm.init(conf.model_axis, conf.norm_eps)
        ffn = FFN.init(conf.model_axis, conf.ff_axis, kf)
        return Block(ln_1, attn, ln_2, ffn)
    @named_call
    def __call__(self, x, mask, sin, cos):
        x = x + self.attn(self.ln_1(x), mask, sin, cos)
        x = x + self.ffn(self.ln_2(x))
        return x

class Aether(eqx.Module, LmHeadModel[AetherConfig], StateDictSerializationMixin):
    conf: AetherConfig = eqx.static_field()
    lm_head: hnn.Linear
    ln_ia: RMSNorm
    ln_ib: RMSNorm
    wia: hnn.Linear
    wib: hnn.Linear
    blocks: Stacked[Block]
    ln_oa: RMSNorm
    ln_ob: RMSNorm
    woa: hnn.Linear
    wob: hnn.Linear
    ln_f: RMSNorm
    mask: hax.NamedArray
    sin: hax.NamedArray
    cos: hax.NamedArray
    @property
    def config(self):
        return self.conf
    @property
    def vocab_size(self):
        return self.conf.vocab_axis.size
    @property
    def Vocab(self):
        return self.conf.vocab_axis
    def resize_vocab(_self, _new_size, _key):
        raise Exception("vocab resize not implemented for Aether")
    @staticmethod
    def init(conf, key):
        klm, kia, kib, koa, kob = jax.random.split(key, 5)
        lm_head = hnn.Linear.init(In=conf.embed_axis, Out=conf.vocab_axis, key=klm, use_bias=False)
        ln_ia = RMSNorm.init(conf.embed_axis, conf.norm_eps)
        ln_ib = RMSNorm.init(conf.embed_axis, conf.norm_eps)
        wia = hnn.Linear.init(In=conf.embed_axis, Out=conf.model_axis, key=kia, use_bias=False)
        wib = hnn.Linear.init(In=conf.embed_axis, Out=conf.model_axis, key=kib, use_bias=False)
        blocks = Stacked.init(
            conf.layer_axis,
            Block,
            gradient_checkpointing=conf.gradient_checkpointing
        )(conf, key=shaped_rng_split(key, conf.layer_count))
        ln_oa = RMSNorm.init(conf.model_axis, conf.norm_eps)
        ln_ob = RMSNorm.init(conf.model_axis, conf.norm_eps)
        woa = hnn.Linear.init(In=conf.model_axis, Out=conf.embed_axis, key=koa, use_bias=False)
        wob = hnn.Linear.init(In=conf.model_axis, Out=conf.embed_axis, key=kob, use_bias=False)
        ln_f = RMSNorm.init(conf.embed_axis, conf.norm_eps)
        mask = precompute_mask(conf.seq_axis, conf.kv_seq_axis)
        sin, cos = precompute_rope(conf.head_axis, conf.seq_axis)
        return Aether(
            conf, lm_head, ln_ia, ln_ib, wia, wib, blocks,
            ln_oa, ln_ob, woa, wob, ln_f, mask, sin, cos
        )
    @named_call
    def __call__(self, xa, xb, attn_mask=None, *, key=None):
        wte = self.lm_head.weight.rearrange((self.conf.vocab_axis, self.conf.embed_axis))
        xa = wte.take(self.conf.vocab_axis, xa)
        xb = wte.take(self.conf.vocab_axis, xb)
        xa = xa.rename({
            self.conf.embed_axis: self.conf.model_axis
        }) + self.wia(self.ln_ia(xa))
        xb = xb.rename({
            self.conf.embed_axis: self.conf.model_axis
        }) + self.wib(self.ln_ib(xb))
        x = (xa + xb) * 0.5
        x = self.blocks.fold(x, self.mask, self.sin, self.cos)
        xa = x.rename({
            self.conf.model_axis: self.conf.embed_axis
        }) + self.woa(self.ln_oa(x))
        xb = x.rename({
            self.conf.model_axis: self.conf.embed_axis
        }) + self.wob(self.ln_ob(x))
        xa = self.lm_head(self.ln_f(xa))
        xb = self.lm_head(self.ln_f(xb))
        return xa, xb
    def compute_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None
    ):
        batch_axis = example.tokens.axes[-2]
        assert (batch_axis.size % 2) == 0
        xa = example.tokens[batch_axis, :batch_axis.size // 2]
        xb = example.tokens[batch_axis, batch_axis.size // 2:]
        pa, pb = self(xa, xb, example.attn_mask, key=key)
        ya, yb = (
            hax.roll(x, -1, axis=self.Pos)
            for x in (xa, xb)
        )
        ya, yb = (
            hax.nn.one_hot(y, self.Vocab, dtype=pa.dtype)
            for y in (ya, yb)
        )
        ma = example.loss_mask[batch_axis, :batch_axis.size // 2]
        mb = example.loss_mask[batch_axis, batch_axis.size // 2:]
        la, lb = (
            hnn.cross_entropy_loss(
                p, self.Vocab, y, reduction,
                reduction_axis=reduction_axis,
                where=m
            )
            for (p, y, m) in ((pa, ya, ma), (pb, yb, mb))
        )
        loss = (la + lb) * 0.5
        return loss