# TODO: Remove.
import jax
import jax.numpy as np
import unittest
import numpy as onp
from jaxwell import fdfd, vecfield

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


class TestJaxwell(unittest.TestCase):
    def setUp(self):
        self.b = onp.zeros((10, 10, 10), onp.complex128)
        self.b[5, 5, 5] = 1.0
        self.b = (0 * self.b, 0 * self.b, self.b)
        self.z = (onp.zeros((10, 10, 10)),) * 3
        self.params = fdfd.Params(pml_ths=((10, 10),) * 3, pml_omega=0.3, max_iters=1)

    def test_solve(self):
        x, err = fdfd.solve(self.params, self.z, self.b)
        self.assertAlmostEqual(err, 35.25115523)

    def test_norm_grad(self):
        def foo(a):
            return np.linalg.norm(a)

        b = onp.zeros((10, 10, 10), onp.complex128)
        grad_x = jax.grad(foo)(b)
        print(grad_x)

    def test_vecfield_grad(self):
        def foo(a):
            return vecfield.norm(a)

        grad_a = jax.grad(foo)(self.b)
        print(grad_a)

    def test_grad(self):
        def foo(z, b):
            res, err = fdfd.solve(self.params, z, b)
            norm = vecfield.norm(res)
            return norm

        grad_z, grad_b = jax.grad(foo, (0, 1))(self.z, self.b)

        # Used as pseudo-golden tests.
        self.assertAlmostEqual(sum(np.sum(g) for g in grad_z), -1903.85547014)
        self.assertAlmostEqual(
            sum(np.sum(g) for g in grad_b), 43.89242676 - 3.725788e-15j
        )

    def test_loop_fns(self):
        x, errs = fdfd.solve_impl(self.z, self.b, params=self.params)
        self.assertEqual(len(x), 3)
        self.assertEqual(x[0].shape, (10, 10, 10))
        self.assertEqual(len(errs), 1)
        self.assertAlmostEqual(errs[0], 35.25115523)

    def test_adjoint(self):
        x, errs = fdfd.solve_impl(self.z, self.b, adjoint=True, params=self.params)
        self.assertAlmostEqual(errs[0], 0.01358992)

    def test_solve_vmap_and_grad(self):
        b0 = self.b
        b1 = tuple(2.0 * component for component in self.b)
        b_batch = tuple(np.stack((c0, c1), axis=0) for c0, c1 in zip(b0, b1))

        solve_vmapped = jax.vmap(lambda b_i: fdfd.solve(self.params, self.z, b_i))

        x_batch, err_batch = solve_vmapped(b_batch)
        x0, err0 = fdfd.solve(self.params, self.z, b0)
        x1, err1 = fdfd.solve(self.params, self.z, b1)

        for xb_i, x0_i, x1_i in zip(x_batch, x0, x1):
            onp.testing.assert_allclose(xb_i[0], x0_i)
            onp.testing.assert_allclose(xb_i[1], x1_i)

        self.assertAlmostEqual(err_batch[0], err0)
        self.assertAlmostEqual(err_batch[1], err1)

        def loss_single(z, b):
            x, _ = fdfd.solve(self.params, z, b)
            return vecfield.norm(x)

        def loss_batch(z, b):
            norms = jax.vmap(lambda b_i: loss_single(z, b_i))(b)
            return np.sum(norms)

        grad_z_batch, grad_b_batch = jax.grad(loss_batch, (0, 1))(self.z, b_batch)
        grad_z_0, grad_b_0 = jax.grad(loss_single, (0, 1))(self.z, b0)
        grad_z_1, grad_b_1 = jax.grad(loss_single, (0, 1))(self.z, b1)

        for gz_b, gz_0, gz_1 in zip(grad_z_batch, grad_z_0, grad_z_1):
            onp.testing.assert_allclose(gz_b, gz_0 + gz_1)

        for gb_b, gb_0, gb_1 in zip(grad_b_batch, grad_b_0, grad_b_1):
            onp.testing.assert_allclose(gb_b[0], gb_0)
            onp.testing.assert_allclose(gb_b[1], gb_1)


if __name__ == "__main__":
    unittest.main()
