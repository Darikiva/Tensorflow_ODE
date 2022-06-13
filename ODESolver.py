from cmath import inf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import SampleModel as sm
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

tf.keras.backend.set_floatx("float64")

N = 500  # number of points to eval


class ODESolver:
    def __init__(
        self, sample_model: sm.SampleModel, nnmodel_hyperparamaters_dict
    ) -> None:
        self.sample_model = sample_model
        self.__create_model(nnmodel_hyperparamaters_dict)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=100, decay_rate=0.99
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def train(
        self, loss_threshold, logging=False, path_to_plots=None):
        bar = tqdm()
        best_model = tf.keras.models.clone_model(self.model)
        min_loss = np.Inf
        iteration_number = 0
        while min_loss > loss_threshold:
            try:
                rand_pts = np.random.uniform(
                    low=self.sample_model.T[0], high=self.sample_model.T[1], size=(N,)
                )
                rand_pts = np.sort(rand_pts)
                for i in range(self.sample_model.dim):
                    rand_pts[self.__get_pos_ind(equation_idx=i, condition_idx=0, points_amount=N)] = self.sample_model.pos[i]
                rand_pts = tf.expand_dims(rand_pts, axis=1)
                loss, start_loss = self.__train_step(rand_pts)
                if (loss) + (start_loss) < min_loss:
                    min_loss = (loss) + start_loss
                    best_model.set_weights(self.model.get_weights())
                if logging:
                    bar.set_description(
                        "Loss: {:.5f} {:.5f} min loss: {:.6f}".format(loss, start_loss, min_loss)
                    )
                    bar.update(1)
                iteration_number += 1
            except KeyboardInterrupt:
                print("ctrl-c")
                break
        self.model = best_model
        return iteration_number

    def draw_plot(self, save_path):
        gx = np.linspace(self.sample_model.T[0], self.sample_model.T[1], N)
        y = self.model(gx)
        plt.plot(gx, np.squeeze(y), "m-", label="Tensorflow")
        if hasattr(self.sample_model, "theoretical") and callable(
            getattr(self.sample_model, "theoretical")
        ):
            print("theoretical")
            plt.plot(
                gx,
                self.sample_model.theoretical(gx).T,
                "b-",
                label="Theoretical",
            )
        if np.all(self.sample_model.pos == self.sample_model.T[0]):
            sol = solve_ivp(
                self.sample_model.equation,
                self.sample_model.T,
                self.sample_model.val[0],
                dense_output=True
            )
            rk = sol.sol(gx)
            print("helo")
            plt.plot(gx, rk.T, "g-", label="Runge-Kutta")
        plt.title(self.sample_model.__doc__)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc="best")
        plt.savefig(save_path)
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


    def __train_step(self, x):
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            dy = [self.model(x)]
            y = self.model(x)
            # if self.sample_model.dim >= 1:
            for i in range(self.sample_model.max_order):
                new_dy = tf.stack(
                    [g.gradient(y[:, i], x) for i in range(self.sample_model.dim)],
                    axis=1,
                )
                new_dy = tf.squeeze(new_dy, axis=2)
                dy.append(new_dy)
            wow1 = self.sample_model.tf_equation(x, *dy)
            rhs = tf.stack(wow1, axis=1)
            # else:
            # dy = g.gradient(y, x)
            # rhs = self.sample_model.tf_equation(x, y)
            loss = None
            for i in range(self.sample_model.equations_amount):
                current_equation_order = self.sample_model.orders[i]
                residual = dy[current_equation_order][i] - rhs[i]
                if not loss:
                    loss = tf.math.square(residual)
                else:
                    loss += tf.math.square(residual)
            residual = dy[0] - rhs
            loss = tf.math.square(residual)
            ind_list = []
            val = []
            for i in range(self.sample_model.equations_amount):
                for j in range(self.sample_model.orders[i]):
                    ind = self.__get_pos_ind(equation_idx=i, condition_idx=j, points_amount=N)
                    ysl = y[ind, i]
                    if [ind, i] in ind_list:
                        val[ind_list.index([ind, i])] += (tf.math.square(ysl - self.sample_model.get_val(equation_idx=i)[0])) * N
                    else:
                        ind_list.append([ind, i])
                        val.append((tf.math.square(ysl - self.sample_model.get_val(equation_idx=i)[0])) * N)
            start_loss = tf.sparse.SparseTensor(
                indices=ind_list, values=val, dense_shape=loss.shape
            )
            start_loss = tf.sparse.to_dense(start_loss)
            total_loss = loss + start_loss
            gradients = g.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            return float(tf.math.reduce_mean(loss)), float(
                tf.math.reduce_mean(start_loss)
            )

    def __get_pos_ind(self, equation_idx, condition_idx, points_amount):
        return round(
            (self.sample_model.get_pos(equation_idx)[condition_idx] - self.sample_model.T[0])
            / (self.sample_model.T[1] - self.sample_model.T[0])
            * (points_amount - 1)
        )

    def __create_model(self, hyperparameters):
        layers_num_str = "layers_num"
        default_neuron_num_str = "default_neuron_num"
        default_activation_func_str = "default_activation_func"
        layers_str = "layers"

        layers_number = hyperparameters[layers_num_str]
        default_neuron_num = hyperparameters[default_neuron_num_str]
        default_activation_func = hyperparameters[default_activation_func_str]

        model = keras.models.Sequential()
        model.add(layers.InputLayer(input_shape=(1,)))

        if layers_num_str in hyperparameters:
            layers_number = hyperparameters[layers_num_str]
        for i in range(layers_number):
            neuronons_num = default_neuron_num
            activation_func = default_activation_func
            str_i = str(i)
            if str_i in hyperparameters[layers_str]:
                neuronons_num = hyperparameters[layers_str][str_i]["neuron_num"]
                activation_func = hyperparameters[layers_str][str_i]["activation_func"]
            model.add(layers.Dense(neuronons_num, activation=activation_func))
        model.add(layers.Dense((self.sample_model.dim)))
        self.model = model
