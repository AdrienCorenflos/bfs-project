from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd, jit, value_and_grad
from jax.experimental.host_callback import id_print
from jax.experimental.optimizers import adam

import tqdm
from .utils import MVNormalParameters
from .ekf import filter_routine
from .operators import smoothing_operator


def make_associative_smoothing_params(transition_function, Qk, i, n, mk, Pk, xk):
    predicate = i == n - 1

    jac_trans = jacfwd(transition_function, 0)

    def _last(_):
        return mk, jnp.zeros_like(Pk), Pk

    def _generic(_):
        return _make_associative_smoothing_params_generic(transition_function, jac_trans, Qk, mk, Pk, xk)

    return lax.cond(predicate,
                    _last,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_smoothing_params_generic(transition_function, jac_transition_function, Qk, mk, Pk, xk):
    F = jac_transition_function(xk)
    Pp = F @ Pk @ F.T + Qk

    E = jlinalg.solve(Pp, F @ Pk, sym_pos=True).T

    g = mk - E @ (transition_function(xk) + F @ (mk - xk))
    L = Pk - E @ Pp @ E.T

    return g, E, L


def smoother_routine(transition_function: Callable,
                     transition_covariance: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearisation_points: jnp.ndarray = None):
    """ Computes the predict-update routine of the Extended Kalman Filter equations
    using temporal parallelization and returns a series of filtered_states TODO:reference

    Parameters
    ----------
    transition_function: callable
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariance for each time step
        observation error covariances for each time step
    filtered_states: MVNormalParameters
        states resulting from (iterated) EKF
    linearisation_points: (n, D) array, optional
        points at which to compute the jacobians, typically previous run.

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = filtered_states.mean.shape[0]

    if linearisation_points is None:
        linearisation_points = filtered_states.mean

    @vmap
    def make_params(i, mk, Pk, xk):
        return make_associative_smoothing_params(transition_function, transition_covariance,
                                                 i, n_observations, mk, Pk, xk)

    gs, Es, Ls = make_params(jnp.arange(n_observations), filtered_states.mean,
                             filtered_states.cov, linearisation_points)

    smoothed_means, _, smoothed_covariances = lax.associative_scan(smoothing_operator, (gs, Es, Ls), reverse=True)

    return vmap(MVNormalParameters)(smoothed_means, smoothed_covariances)


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariance: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariance: jnp.ndarray,
                              initial_linearization_states: MVNormalParameters = None,
                              n_iter: int = 100):
    """
    Computes the Gauss-Newton iterated extended Kalman smoother

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K, K)  array
        observation error covariances for each time step, if passed only one, it is repeated n times
    initial_linearization_states: (N, D) array, optional
        points at which to compute the jacobians durning the first pass.
    n_iter: int
        number of times the filter-smoother routine is computed


    Returns
    -------
    iterated_smoothed_trajectories: MVNormalParameters
        The result of the smoothing routine

    """

    @jit
    def body(linearization_points, _):
        if linearization_points is not None:
            linearization_points = linearization_points.mean
        filtered_states, _ = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                            observation_function, observation_covariance, linearization_points)
        return smoother_routine(transition_function, transition_covariance, filtered_states,
                                linearization_points), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return iterated_smoothed_trajectories


def max_likelihood_smoother_routine(initial_state: MVNormalParameters,
                                    observations: jnp.ndarray,
                                    transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                    transition_covariance: jnp.ndarray,
                                    observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                    observation_covariance: jnp.ndarray,
                                    initial_linearization_states: MVNormalParameters = None,
                                    method: callable = adam,
                                    reg: float = 1e-2,
                                    step_size: float = 1e-3,
                                    num_steps: int = 150,
                                    regularization_function: callable = lambda z: 0

                                    ):
    """
    Computes the Gauss-Newton iterated extended Kalman smoother

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K, K)  array
        observation error covariances for each time step, if passed only one, it is repeated n times
    initial_linearization_states: (N, D) array, optional
        points at which to compute the jacobians durning the first pass.
    method: callable
        minimization method
    reg: float
        Regularization to prevent successive linearization points to fall too far from each other.
    step_size: float
        Step size for optimizer.
    num_steps: float
        Number of optimizing steps.
    regularization_function: Callable
        Function used in the loss to regularize the result

    Returns
    -------

    iterated_smoothed_trajectories: MVNormalParameters
        The result of the smoothing routine
    final linearization_points: jnp.ndarray
        The optimized linearization points

    """
    T = observations.shape[0]
    dx = initial_state.mean.shape[0]

    def _loss(linearization_points):
        _filtered_states, ll = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                              observation_function, observation_covariance,
                                              linearization_points)

        return -ll + reg * regularization_function(linearization_points)

    if initial_linearization_states is None:
        initial_linearization_points = jnp.zeros((T, dx), dtype=observations.dtype)
    else:
        initial_linearization_points = initial_linearization_states.mean

    init_fun, update_fun, get_params = method(step_size)
    opt_state = init_fun(initial_linearization_points)

    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(_loss)(get_params(opt_state))
        grads = jnp.nan_to_num(grads)
        opt_state = update_fun(step, grads, opt_state)
        return value, opt_state, grads

    progress_bar = tqdm.trange(num_steps, desc="Current loss: ")
    for i in progress_bar:
        value, opt_state, grads = step(i, opt_state)
        progress_bar.set_description(f"Current loss: {float(value):.2f}, Max grad: {float(jnp.abs(grads).max()):.2f}")
        progress_bar.refresh()

    res = get_params(opt_state)
    filtered_states, ll = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                         observation_function, observation_covariance, res)

    print("Final log-lik: ", ll)
    return smoother_routine(transition_function, transition_covariance, filtered_states, res), res
