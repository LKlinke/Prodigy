from typing import Union

import sympy
import logging
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from .forward.distribution import Distribution
from .forward.exceptions import ParameterError
from probably.pgcl import VarExpr
from probably.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class Plotter:
    """ Plotter that generates histogram plots using matplotlib."""

    @staticmethod
    def _create_2d_hist(function: Distribution, var_1: str, var_2: str, threshold: Union[str, int]):

        x = sympy.S(var_1)
        y = sympy.S(var_2)

        # Marginalize distribution to the variables of interest.
        marginal = function.marginal(x, y)
        marginal = marginal.set_variables(*function.get_variables())
        logger.debug(f"Creating Histogram for {marginal}")
        # Collect relevant data from the distribution and plot it.
        if marginal.is_finite():
            coord_and_prob = dict()
            maxima = {x: 0, y: 0}
            max_prob = 0
            colors = []

            # collect the coordinates and probabilities. Also compute maxima of probabilities and degrees
            terms = 0
            prob_sum = 0
            for prob, state in marginal:
                if isinstance(threshold, str) and prob_sum >= sympy.S(threshold):
                    break
                if isinstance(threshold, int) and terms >= threshold:
                    break
                s_prob = sympy.S(prob)
                maxima[x], maxima[y] = max(maxima[x], state[var_1]), max(maxima[y], state[var_2])
                coord = (state[var_1], state[var_2])
                coord_and_prob[coord] = s_prob
                max_prob = max(s_prob, max_prob)
                terms += 1
                prob_sum += s_prob

            # Zero out the colors array
            for _ in range(maxima[y] + 1):
                colors.append(list(0.0 for _ in range(maxima[x] + 1)))

            # Fill the colors array with the previously collected data.
            for coord in coord_and_prob:
                colors[coord[1]][coord[0]] = float(coord_and_prob[coord])

            # Plot the colors array
            c = plt.imshow(colors, vmin=0, origin='lower', interpolation='nearest', cmap="turbo", aspect='auto')
            plt.colorbar(c)
            plt.gca().set_xlabel(f"{x}")
            plt.gca().set_xticks(range(0, maxima[x] + 1))
            plt.gca().set_ylabel(f"{y}")
            plt.gca().set_yticks(range(0, maxima[y] + 1))
            plt.show()
        else:
            # make the marginal finite.
            plt.ion()
            prev_sum = marginal
            for subsum in marginal.approximate(threshold):
                if subsum == prev_sum:
                    continue
                Plotter._create_2d_hist(subsum, var_1, var_2, threshold)
                prev_sum = subsum

    @staticmethod
    def _create_histogram_for_variable(function: Distribution, var: Union[str, VarExpr], threshold: Union[str, int]) -> None:
        marginal = function.marginal(var)
        if marginal.is_finite():
            data = []
            ind = []
            terms = prob_sum = 0
            for prob, state in marginal:
                if isinstance(threshold, int) and terms >= threshold:
                    break
                if isinstance(threshold, str) and prob_sum > sympy.S(threshold):
                    break
                data.append(float(sympy.S(prob)))
                ind.append(float(state[var]))
                prob_sum += sympy.S(prob)
                terms += 1
            ax = plt.subplot()
            my_cmap = plt.cm.get_cmap("Blues")
            colors = my_cmap([x / max(data) for x in data])
            sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, max(data)))
            sm.set_array([])
            ax.bar(ind, data, 1, linewidth=.5, ec=(0, 0, 0), color=colors)
            ax.set_xlabel(f"{var}")
            ax.set_xticks(ind)
            ax.set_ylabel(f'Probability p({var})')
            plt.get_current_fig_manager().set_window_title("Histogram Plot")
            plt.gcf().suptitle("Histogram")
            plt.colorbar(sm)
            plt.show()
        else:
            prev_gf = marginal
            for gf in marginal.approximate(threshold):
                if not gf.is_zero_dist() and not prev_gf == gf:
                    Plotter._create_histogram_for_variable(gf, var, threshold)
                prev_gf = gf

    @staticmethod
    def plot(function: Distribution, *variables: Union[str, sympy.Symbol], threshold: Union[str, int]) -> None:
        """ Shows the histogram of the marginal distribution of the specified variable(s). """
        if function.get_parameters():
            raise Exception("Cannot Plot parametrized functions.")
        if variables:
            if len(variables) > 2:
                raise ParameterError(f"create_plot() cannot handle more than two variables!")
            if len(variables) == 2:
                Plotter._create_2d_hist(function, var_1=variables[0], var_2=variables[1], threshold=threshold)
            if len(variables) == 1:
                Plotter._create_histogram_for_variable(function, var=variables[0], threshold=threshold)
        else:
            if len(function.get_variables()) > 2:
                raise Exception("Multivariate distributions need to specify the variable to plot")

            elif len(function.get_variables()) == 2:
                vars = list(function.get_variables())
                Plotter._create_2d_hist(function, var_1=vars[0], var_2=vars[1], threshold=threshold)
            else:
                for var in function.get_variables():
                    Plotter._create_histogram_for_variable(var, threshold=threshold)