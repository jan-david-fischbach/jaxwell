# %%
# ruff: noqa: E402
# %%
# This is needed to enable JAX's double-precision mode, see
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision
# for additional info.
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jaxwell
import matplotlib.pyplot as plt
import numpy as onp
import treams as tr

# import treams

# Check to make sure double-precision is enabled.
assert np.zeros((1,), np.float64).dtype == np.float64


# %%
# Build the structure, source, and loss sub-models.
def plot_field(field, mask=True, vmax=0.1):
    if mask:
        field = [onp.where(eps_sphere[0] < 3, s, onp.nan) for s in field]
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(131 + i)
        plt.pcolormesh(
            positions[0][:, :, 0],
            positions[1][:, :, 0],
            onp.abs(field[i][:, :, num_pixels // 2]),
            vmin=0,
            vmax=vmax,
        )
        plt.axis("off")


def center_px(shape):
    return np.array([s / 2 - 0.5 for s in shape])


def center_fl(shape, dx):
    return dx * center_px(shape)


def structure(radius, shape, center):
    """Builds a ball of radius `radius`

    For simplicity, we do not take into account the offsets between the x-, y-,
    and z-components of epsilon in the Yee cell.

    Args:
      radius: radius of the spehere in pixels.
      shape: `(xx, yy, zz)` tuple defining the shape of the simulation.
      center: specifies the center of the sphere as a tuple.
        If `None` the sphere is centered in the simulation region
    """
    if center is None:
        center = center_px(shape)

    center = np.array(center).reshape((-1,) + (1,) * 3)
    arr = np.linalg.norm(np.indices(shape) - center, axis=0)
    return arr <= radius
    ((arr <= radius - 1) * 0.25 + (arr <= radius) * 0.5 + (arr <= radius + 1) * 0.25)


eps_bg = 2
eps_fg = 12
dx = 40
wlen = 1550 / dx
k0 = omega = 2 * onp.pi / wlen
basis = tr.SphericalWaveBasis.default(3)
num_pixels = 80
R = num_pixels / 2 / 2  # Radius of the sphere
print(f"Radius: {R}")
shape = (num_pixels,) * 3


def to_spherical_coordinates(x, y, z, k0, center):
    """Converts to spherical coordinates in units of k0"""
    x, y, z = (comp - cent for comp, cent in zip([x, y, z], center))
    r = onp.sqrt(x**2 + y**2 + z**2)
    theta = onp.arccos(z / r)
    phi = onp.sign(y) * onp.arccos(x / onp.sqrt(x**2 + y**2))
    theta[r == 0] = 0
    phi[r == 0] = 0
    return r * k0, theta, phi


# %%
x = y = z = np.arange(num_pixels) * dx
cent = center_fl(shape, dx)
cent = cent.at[0].set(cent[0] + 200)

positions = onp.meshgrid(x, y, z, indexing="ij")
positions_spherical = to_spherical_coordinates(*positions, k0 / dx, cent)

params = jaxwell.Params(
    pml_ths=((10, 10), (10, 10), (10, 10)),
    pml_omega=omega,
    eps=1e-6,
    max_iters=int(1e6),
)

sphere = onp.array(structure(R, (num_pixels,) * 3, center=cent / dx))
eps_sphere = [
    sphere * (eps_fg - eps_bg) + eps_bg
] * 3  # not super accurate (do subpixel smoothing)
mode = basis[6]
field_inc = tr.special.vsw_rA(mode[1], mode[2], *positions_spherical, mode[3])
field_inc = onp.moveaxis(field_inc, -1, 0)

# %%
plot_field(field_inc)
# %%
plot_field(eps_sphere, mask=False, vmax=None)

# %%
b = onp.array([-(omega**2) * (eps_sphere[0] - eps_bg)] * 3) * field_inc
b = tuple(b)
z = tuple([omega**2 * eps for eps in eps_sphere])

# %%
plot_field(b, mask=False)


# %%
def call_back(x, errs):
    jax.debug.print("{err}", err=onp.sum(errs))


field_scat, err = jaxwell.fdfd.solve_impl(
    z, b, params=params, monitor_every_n=1, monitor_fn=call_back
)

# %%
plot_field(field_scat)
