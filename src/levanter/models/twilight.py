import dataclasses
from dataclasses import dataclass
import equinox as eqx
import jax
import haliax as hax
import haliax.nn as hnn
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.compat.torch_serialization import StateDictSerializationMixin
from levanter.models.lm_model import LmConfig, LmHeadModel

@LmConfig.register_subclass("twilight")
@dataclass(frozen=True)
class TwilightConfig(LmConfig):
    model_dim: int
    ff_dim: int
    layer_count: int
    norm_eps: float
    seq_len: int
    vocab_size: int
    gradient_checkpointing: bool

    seq_axis = property(lambda self: hax.Axis(name="seq", size=self.seq_len))
    model_axis = property(lambda self: hax.Axis(name="model", size=self.model_dim))
    hidden_axis = property(lambda self: self.model_axis.alias("hidden"))
    ff_axis = property(lambda self: hax.Axis(name="ff", size=self.ff_dim))
    layer_axis = property(lambda self: hax.Axis(name="layer", size=self.layer_count))
    head_axis = property(lambda self: hax.Axis(name="head", size=self.model_dim // self.query_count))
    vocab_axis = property(lambda self: hax.Axis(name="vocab", size=self.vocab_size))
    
    @property
    def model_type(cls):
        return Twilight
    @property
    def KeyPos(self):
        return self.kv_seq_axis
    @property
    def Pos(self):
        return self.seq_axis
    def build(self, Vocab, *, key):
        return Twilight.init(self, key)

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

class Recurrence(eqx.Module, StateDictSerializationMixin):
    conf: TwilightConfig = eqx.static_field()
    wr: hnn.Linear
    wf: hnn.Linear
    wc: hnn.Linear
    wo: hnn.Linear
    ln: RMSNorm
    @staticmethod
    def init(conf, key):
        kr, kf, kc, ko = jax.random.split(key, 4)
        wr = hnn.Linear.init(In=conf.model_axis, Out=conf.hidden_axis, key=kr, use_bias=False)
        wf = hnn.Linear.init(In=conf.model_axis, Out=conf.hidden_axis, key=kf, use_bias=False)
        wc = hnn.Linear.init(In=conf.model_axis, Out=conf.hidden_axis, key=kc, use_bias=False)
        wo = hnn.Linear.init(In=conf.hidden_axis, Out=conf.model_axis, key=ko, use_bias=False)
        ln = RMSNorm.init(conf.hidden_axis, conf.norm_eps)
        return Recurrence(conf, wr, wf, wc, wo, ln)
    @named_call
    def __call__(self, x):
        r, f, c = self.wr(x), self.wf(x), self.wc(x)
        r = hnn.silu(r)
        f = hnn.sigmoid(f)
        c = hnn.silu(c)
        def recurrence(a, b):
            fa, ca = a
            fb, cb = b
            return fa * fb, fb * ca + (1 - fb) * cb
        _, h = jax.lax.associative_scan(recurrence, (f.array, c.array), axis=1)
        h = hax.NamedArray(h, c.axes)
        y = self.wo(r * self.ln(h))
        return y

class Block(eqx.Module, StateDictSerializationMixin):
    ln_1: RMSNorm
    rec: Recurrence
    ln_2: RMSNorm
    ffn: FFN
    @staticmethod
    def init(conf, key):
        kr, kf = jax.random.split(key, 2)
        ln_1 = RMSNorm.init(conf.model_axis, conf.norm_eps)
        rec = Recurrence.init(conf, kr)
        ln_2 = RMSNorm.init(conf.model_axis, conf.norm_eps)
        ffn = FFN.init(conf.model_axis, conf.ff_axis, kf)
        return Block(ln_1, rec, ln_2, ffn)
    @named_call
    def __call__(self, x):
        x = x + self.rec(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class Twilight(eqx.Module, LmHeadModel[TwilightConfig], StateDictSerializationMixin):
    conf: TwilightConfig = eqx.static_field()
    ln_p: RMSNorm
    blocks: Stacked[Block]
    ln_f: RMSNorm
    lm_head: hnn.Linear
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
        raise Exception("vocab resize not implemented for Twilight")
    @staticmethod
    def init(conf, key):
        kw, kl = jax.random.split(key, 2)
        ln_p = RMSNorm.init(conf.model_axis, conf.norm_eps)
        blocks = Stacked.init(
            conf.layer_axis,
            Block,
            gradient_checkpointing=conf.gradient_checkpointing
        )(conf, key=shaped_rng_split(key, conf.layer_count))
        ln_f = RMSNorm.init(conf.model_axis, conf.norm_eps)
        lm_head = hnn.Linear.init(In=conf.model_axis, Out=conf.vocab_axis, key=kl, use_bias=False)
        return Twilight(conf, ln_p, blocks, ln_f, lm_head)
    @named_call
    def __call__(self, x, attn_mask=None, *, key=None):
        wte = self.lm_head.weight.rearrange((self.conf.vocab_axis, self.conf.model_axis))
        x = wte.take(self.conf.vocab_axis, x)
        x = self.ln_p(x)
        x = self.blocks.fold(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x