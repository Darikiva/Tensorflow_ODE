import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import SampleModel as sm
from ODESolver import ODESolver
import json
import time
import scipy.interpolate as interp
import os

N = 5000  # number of points to eval


def func(t, z):
    return  [np.sin(t)/(1 + t + z)]


def theoretical(t):
    return np.array(np.vstack((1 / np.power(t, 2), -2 / np.power(t, 3))))


# def func_tf(t, z, dz):
#     return [z[:, 0] + 2*z[:, 1] - dz[:, 0], z[:, 0] - 5 * tf.squeeze(tf.sin(t)) - dz[:, 1]]

# 804


def func_tf(t, z, dz = None):
    squeezed_t = tf.squeeze(t)
    return tf.math.divide(tf.math.sin(squeezed_t),(1 + z + squeezed_t))


def test_for_training_duration(sample_model: sm.SampleModel, save_path):
    path_to_config_file = "/home/divashyn/univer/diploma/Tensorflow_ODE/config.json"
    config_data = None
    with open(path_to_config_file, "r") as config_file:
        config_data = json.load(config_file)
    x_layers_number = []
    y_training_time = []
    for layers_number in range(2, 20, 2):
        config_data["layers_num"] = layers_number
        ode_solver = ODESolver(
            sample_model=sample_model, nnmodel_hyperparamaters_dict=config_data
        )
        start_time = time.time()
        ode_solver.train(config_data["loss_threshold"], logging=True)
        end_time = time.time()
        x_layers_number.append(layers_number)
        y_training_time.append(end_time - start_time)
        ode_solver.draw_plot(save_path + str(layers_number) + ".png")
    plt.plot(x_layers_number, y_training_time, label="time in seconds")
    plt.savefig(save_path + "result.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def test_for_activation_function(sample_model: sm.SampleModel, save_path):
    activation_functions = [
        # "sigmoid",
        # "swish",
        # "softmax",
        # "selu",
        "elu",
        # "hard_sigmoid",
        # "softplus",
        # "tanh",
    ]
    path_to_config_file = "/home/divashyn/univer/diploma/Tensorflow_ODE/config.json"
    config_data = None
    with open(path_to_config_file, "r") as config_file:
        config_data = json.load(config_file)
    x_activation_functions = []
    y_training_time = []
    y2_iteration_numbers = []
    for activation_function in activation_functions:
        x_activation_functions.append(activation_function)
        training_time_sum = 0
        iteration_number_sum = 0
        # for _ in range(10):
        config_data["default_activation_func"] = activation_function
        ode_solver = ODESolver(
            sample_model=sample_model, nnmodel_hyperparamaters_dict=config_data
        )
        start_time = time.time()
        iteration_number = ode_solver.train(config_data["loss_threshold"], logging=True)
        end_time = time.time()
        training_time_sum += end_time - start_time
        iteration_number_sum += iteration_number
        ode_solver.draw_plot(save_path + str(activation_function) + ".png")
        y_training_time.append(training_time_sum / 10.0)
        y2_iteration_numbers.append(iteration_number_sum / 10.0)
    # plt.bar(x_activation_functions, y_training_time)
    # plt.savefig(save_path + "result_time.png")
    # plt.figure().clear()
    # plt.close()
    # plt.cla()
    # plt.clf()

    # plt.bar(x_activation_functions, y2_iteration_numbers)
    # plt.savefig(save_path + "result_iterations.png")
    # plt.figure().clear()
    # plt.close()
    # plt.cla()
    # plt.clf()


def test_layers_and_neuron_numbers(
    sample_model: sm.SampleModel, save_path, config=None
):
    path_to_config_file = "/home/divashyn/univer/diploma/Tensorflow_ODE/config.json"
    if config is None:
        with open(path_to_config_file, "r") as config_file:
            config_data = json.load(config_file)
    else:
        config_data = config.copy()
    x_layers_number = []
    y_neuron_number = []
    z_training_time = []
    for layers_number in range(6, 18, 2):
        config_data["layers_num"] = layers_number
        for neuron_number in range(8, 25, 4):
            config_data["default_neuron_num"] = neuron_number
            ode_solver = ODESolver(
                sample_model=sample_model, nnmodel_hyperparamaters_dict=config_data
            )
            start_time = time.time()
            ode_solver.train(config_data["loss_threshold"], logging=True)
            end_time = time.time()
            x_layers_number.append(layers_number)
            y_neuron_number.append(neuron_number)
            z_training_time.append(end_time - start_time)
            ode_solver.draw_plot(
                os.path.join(
                    save_path, str(layers_number) + "_" + str(neuron_number) + ".png"
                )
            )
    plotx, ploty = np.meshgrid(
        np.linspace(np.min(x_layers_number), np.max(x_layers_number), 10),
        np.linspace(np.min(y_neuron_number), np.max(y_neuron_number), 10),
    )
    plotz = interp.griddata(
        (x_layers_number, y_neuron_number),
        z_training_time,
        (plotx, ploty),
        method="linear",
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Layers number")
    ax.set_ylabel("Neurons number")
    ax.set_zlabel("Time, seconds")
    ax.plot_surface(plotx, ploty, plotz, cstride=1, rstride=1, cmap="viridis")
    plt.savefig(os.path.join(save_path, "result.png"))
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    textfile = open(os.path.join(save_path, "a_file.txt"), "w")
    for el in x_layers_number:
        textfile.write(str(el))
        textfile.write(" ")
    textfile.write("\n")
    for el in y_neuron_number:
        textfile.write(str(el))
        textfile.write(" ")
    textfile.write("\n")
    for el in z_training_time:
        textfile.write(str(el))
        textfile.write(" ")
    textfile.write("\n")


def test_different_activation_functions_for_layers_and_neuron_number(
    sample_model: sm.SampleModel, save_path
):
    activation_functions = [
        "sigmoid",
        "swish",
        # "softmax",
        "selu",
        "elu",
        "hard_sigmoid",
        "softplus",
        "tanh",
    ]
    path_to_config_file = "/home/divashyn/univer/diploma/Tensorflow_ODE/config.json"
    with open(path_to_config_file, "r") as config_file:
        config_data = json.load(config_file)
    for activation_function in activation_functions:
        new_save_path = os.path.join(save_path, activation_function)
        config_data["default_activation_func"] = activation_function
        test_layers_and_neuron_numbers(sample_model, new_save_path, config_data)


def simple_test(sample_model: sm.SampleModel, save_path):
    path_to_config_file = "/home/divashyn/univer/diploma/Tensorflow_ODE/config.json"
    with open(path_to_config_file, "r") as config_file:
        config_data = json.load(config_file)
    ode_solver = ODESolver(
        sample_model=sample_model, nnmodel_hyperparamaters_dict=config_data
    )
    ode_solver.train(config_data["loss_threshold"], logging=True)
    ode_solver.draw_plot(os.path.join(save_path, "result.png"))


if __name__ == "__main__":
    sample_model = sm.SampleModel(
        func=func,
        func_tf=func_tf,
        pos=np.array([[0.0]]),
        val=np.array([[0.0]]),
        interval=np.array([0, 10]),
        # theoretical=theoretical,
    )
    save_path = "/home/divashyn/univer/diploma/Tensorflow_ODE/"
    simple_test(sample_model, save_path)
    # test_different_activation_functions_for_layers_and_neuron_number(sample_model=sample_model, save_path=save_path)
