import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class MAML:
    def __init__(self):
        self.batch_size = 5
        self.no_tasks = 5
        self.parameters = np.array([-10., 10.])
        self.goal_parameters = ([5., -5.])
        self.x_val, self.y_val = self.generate_data(self.goal_parameters, 100, noise=False)

        self.xlim = self.ylim = [-10., 10]

        self.noise_mean = 0
        self.noise_sigma = 0.3

        self.beta = 0.0003
        self.alpha = 10 * self.beta
        self.gamma = 10 * self.beta

        self.plot_history = []
        self.max_plot_history = 3
        self.iteration = 0

    def sample_batch_of_tasks(self):
        tasks_params = []
        for i in range(self.no_tasks):
            a = self.goal_parameters[0]
            b = np.random.uniform(0, 10)
            tasks_params.append((a, b))
        return tasks_params

    def generate_data(self, parameters, batch_size, noise=True):
        x = np.random.uniform(-2.5, 2.5, batch_size)
        if noise:
            x += self._noise(batch_size)
        y = self.get_y(x, parameters)
        return x, y

    @staticmethod
    def get_y(x, parameters):
        return parameters[0] * x**3 + 5*parameters[1] * x**2

    def _noise(self, batch_size):
        return np.random.normal(self.noise_mean, self.noise_sigma, batch_size)

    @staticmethod
    def derivative_loss(x, y, params):
        derivative_loss = np.array([params[0] * x**6 + 5 * params[1] * x**5 - x**3 * y,
                                    5 * params[0] * x**5 + 25 * params[1] * x**4 - 5 * x**2 * y])

        return np.mean(derivative_loss, axis=1)

    def loss(self, x, y, params):
        loss = 0.5 * (y - self.get_y(x, params))**2
        return np.mean(loss)

    def plot(self, batch_of_tasks, text):

        #### TRACK GRADIENT HISTORY ####
        if len(self.plot_history) > self.max_plot_history:
            self.plot_history.pop(0)
        self.plot_history.append(self.parameters)

        #### PLOT SURFACE ####
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = Y = np.arange(self.xlim[0], self.ylim[1], 0.25)
        Z = []
        for y in X:
            for x in Y:
                Z.append(self.loss(self.x_val, self.y_val, np.array([x, y])))
        Z = np.array(Z)
        Z = Z.reshape(int(np.sqrt(Z.shape[0])),
                      int(np.sqrt(Z.shape[0])))
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)


        #### PLOT START AND END POINT ####
        ax.scatter(xs=[self.goal_parameters[0]],
                   ys=[self.goal_parameters[1]],
                   zs=[self.loss(self.x_val, self.y_val, self.goal_parameters)], c='red',
                   s=50)

        for i, a in enumerate(self.plot_history):
            i = i+1
            ax.scatter(xs=[a[0]], ys=[a[1]], zs=[self.loss(self.x_val, self.y_val, a)], c='black', s=50,
                       alpha=float(i/4))

        #### PLOT SUBTASKS PARAMETERS ####
        for i, task in enumerate(batch_of_tasks):
            ax.scatter(xs=[task[0]], ys=[task[1]], zs=[self.loss(self.x_val, self.y_val, task)], marker="*",
                       c='purple', s=50)

        ax.set_title(text)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel('Parameter a')
        ax.set_ylabel('Parameter b')
        ax.set_zlabel('Loss')
        plt.draw()
        plt.savefig("/Users/damian/Desktop/gif/{}.png".format(self.iteration))
        self.iteration += 1
        plt.pause(0.0001)
        plt.clf()
        plt.close()

    def run(self):
        batch_of_tasks = self.sample_batch_of_tasks()

        ### META-LEARNING ###
        for i in range(60):
            task_gradients = []
            for task in batch_of_tasks:
                x, y = self.generate_data(task, self.batch_size)
                task_parameters = self.parameters - self.alpha * self.derivative_loss(x, y, self.parameters)
                x, y = self.generate_data(task, self.batch_size)
                task_gradient = self.derivative_loss(x, y, task_parameters)
                task_gradients.append(task_gradient)
            task_gradients = np.array(task_gradients)
            meta_gradient = np.sum(task_gradients, axis=0)
            self.parameters = self.parameters - self.beta * meta_gradient
            self.plot(batch_of_tasks, "First step: meta-training...")

        ### FINE_TUNING ###
        x, y = self.generate_data(self.goal_parameters, self.batch_size)
        while True:
            self.parameters = self.parameters - self.gamma * self.derivative_loss(x, y, self.parameters)
            if np.linalg.norm(self.parameters-self.goal_parameters) < 0.1:
                break
            self.plot(batch_of_tasks, "Second step: fine-tuning...")


maml = MAML()
maml.run()
