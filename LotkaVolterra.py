
class LotkaVolterra(object):
'''The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The populations change through time according to the pair of equations:

where

x is the number of prey (for example, rabbits);
y is the number of some predator (for example, foxes);

{\displaystyle {\tfrac {dy}{dt}}}{\tfrac {dy}{dt}} and {\displaystyle {\tfrac {dx}{dt}}}{\tfrac {dx}{dt}} represent the instantaneous growth rates of the two populations;
t represents time;

α, β, γ, δ are positive real parameters describing the interaction of the two species.

The Lotka–Volterra system of equations is an example of a Kolmogorov model,[1][2][3] which is a more general framework that can model the dynamics of ecological systems with predator–prey interactions, competition, disease, and mutualism.'''

    def _init_(self,α,β,δ,γ)

    def dx_dt(self,x,y):
        "Returns the instantaneous growth rate of population x wrt time"
        return self.α*x-self.β*x*y

    def dy_dt(self,x,y):
        "Returns the instantaneous growth rate of population y wrt time "
        return self.δ*x*y-self.γ*y
