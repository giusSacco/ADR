# A Perturbative Approach to the Frequency Response

## A brief recap of the constant wind

At the steady-state we must solve 

$$
D \nabla^2 c_{ss}(x) - \vec{v} \cdot \vec{\nabla} c_{ss}(x) + a(x) c_{ss}(x) - b c^2_{ss}(x)= 0
$$

Now, let us define the chemistry as

$$
a(x) = a_0 + \delta a(x)
$$

with, formally, __a small perturbation__ $\delta a(x)$, and $\langle\delta a(x)\rangle_x =0$.

Now, we have the following useful __adimensional__ quantities:

$$
\vec{u} = \frac{\vec{v}}{2 \sqrt{D a_0}} = \frac{\vec{v}}{v_F}
$$
$$
\xi(x) = \frac{\delta a (x)}{a_0}
$$
$$
\tilde{x} = x \sqrt{\frac{a_0}{D}}
$$

Now, we expect the average concentration to be

$$
c_0 = \frac{a_0}{b}
$$

which is the logistic capacity of the system.
It is therefore smarter to define
$$
\tilde{c}(\tilde{x}) = \frac{c_{ss}(\tilde{x}) - c_0}{c_0}
$$

__At first order in $\delta a$__, the population is the solution to the equation

$$
(-\nabla^2 + \vec{u} \cdot \vec{\nabla} + 1) \tilde{c}(\tilde{x}) = \xi(x)
$$

As we all remember dalle scuole elementtari (cit. che i falzi Unipi non capiranno), the solution formally reads

$$
\tilde{c}(\tilde{x}) = \int G(\tilde{x} - \tilde{x}') \xi(\tilde{x}') d\tilde{x}'
$$

Where $G(\tilde{x})$ is the Green's function of the operator $(-\nabla^2 + \vec{u} \cdot \vec{\nabla} + 1)$. Via [magic](https://www.sciencedirect.com/science/article/abs/pii/S0378437116305830) we find that this reads (in 2 dimensions)

$$
G(\tilde{x}) = \mathcal{C} e^{\vec{u}\cdot \tilde{x}} K_0(\sqrt{1 + |\vec{u}|^2}|\tilde{x}|)
$$

($\mathcal{C}$) is a normalization constant, and $K_0$ is the modified Bessel function of the second kind.

## Sinusoidal perturbation

Let us now perturbe the system with a sinusoidal perturbation. 
In particular, we will consider

$$
\vec{u} = \vec{u}_0 + \delta \vec{u} e^{ i \omega t}
$$

and let's look for a solution of the form

$$
\tilde{c}(\tilde{x}, t) = \tilde{c}_0(\tilde{x}) + \delta \tilde{c}(\tilde{x}) e^{i \omega t}
$$

This, __at first order in $\delta \vec{u}$__, is the solution to the equation

$$
(-\nabla^2 + \vec{u}_0 \cdot \vec{\nabla} + 1 - i \omega) \delta \tilde{c}(\tilde{x}) =  \delta \vec{u} \cdot \vec{\nabla} \tilde{c}_0(\tilde{x})
$$

We can now borrow one from Succi's book and write the solution as

$$
\delta \tilde{c}(\tilde{x}) = \int H_\omega(\tilde{x} - \tilde{x}') \delta \vec{u} \cdot \vec{\nabla} \tilde{c}_0(\tilde{x}') d\tilde{x}'
$$