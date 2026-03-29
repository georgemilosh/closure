import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

data = np.load("Data/sharing_simulation_data/larger_v/2024-10-07_TS_alpha_0-01_drift_3_T_0-1_bal_0-4_dt_3e-4_larger_v_space/2024-10-07_TS_alpha_0-01_drift_3_T_0-1_bal_0-4_dt_3e-4_larger_v_space_moments.npy")

n = data[:,55000:,0]
u = data[:,55000:,1]
P = data[:,55000:,2]
Q = data[:,55000:,3]
E = data[:,55000:,4]
T = data[:,55000:,5]

dx = 0.1227184630308513
dt = 0.0003067961575770996

print("data loaded")

def power(x1, x2):
    # Prevent invalid operations by using a safe implementation
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.where((x1 < 0) & (x2 % 1 != 0), 1, np.power(x1, x2))  # Handle negatives
        result = np.where(np.isfinite(result), result, 1)  # Replace non-finite results with 1
    return result

power_function = make_function(function=power, name='pow', arity=2)

def spatial_derivative(X):
    return np.gradient(X, dx, axis=0, edge_order=2)

def time_derivative(X):
    return np.gradient(X, dt, axis=1, edge_order=2)

dndt = time_derivative(n)
dndx = spatial_derivative(n)
dudt = time_derivative(u)
dudx = spatial_derivative(u)
dpdt = time_derivative(P)
dpdx = spatial_derivative(P)
dqdt = time_derivative(Q)
dqdx = spatial_derivative(Q)
dedt = time_derivative(E)
dedx = spatial_derivative(E)
dtdt = time_derivative(T)
dtdx = spatial_derivative(T)
j = n*u
djdt = time_derivative(j)
djdx = spatial_derivative(j)

#X = np.column_stack((n.flatten(), u.flatten(), P.flatten(), E.flatten(), T.flatten(), dndt.flatten(), dndx.flatten(), dudt.flatten(), dudx.flatten(), dpdt.flatten(), dpdx.flatten(), dqdt.flatten(), dqdx.flatten(), dedt.flatten(), dedx.flatten(),dtdt.flatten(), dtdx.flatten()))
X = np.column_stack((n.flatten(), u.flatten(), E.flatten(), dndt.flatten(), dndx.flatten(), dudt.flatten(), dudx.flatten(), dedt.flatten(), dedx.flatten()))
Y = P.flatten()
sample = np.random.choice(range(len(Y)), int(len(Y)/100), replace=False)
Y_train = Y[sample]
X_train = X[sample]

function_set = ["mul", "add", "sub", "neg", "div"] #power_function

est_gp = SymbolicRegressor(function_set=function_set,generations=1, population_size=5000,
                               stopping_criteria=0.1, p_crossover=0.65, p_subtree_mutation=0.15,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.5, verbose=1, random_state=0, parsimony_coefficient=0.05)
est_gp.fit(X_train, Y_train)
mse = []
for gen in range(200):
    est_gp = est_gp.set_params(generations=gen+2, warm_start=True)
    est_gp.fit(X_train, Y_train)
    mse.append(est_gp._program.raw_fitness_)
    plt.plot(mse)
    plt.title("MSE for the pressure tensor")
    plt.xlabel("Generations")#wnat j, jt, jx, u, n, ux, Px, Ex
    plt.ylabel("MSE")
    plt.show()
    print(est_gp._program)
