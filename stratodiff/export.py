"""Write generated weather as a balloon-sim record (M7): u, v on (time, level, lat, lon), plus pressure and
height per level so simulators that use either can read it."""


def to_balloon_sim(*args, **kwargs):
    raise NotImplementedError("M7: export, after time tiling (SPEC.md)")
