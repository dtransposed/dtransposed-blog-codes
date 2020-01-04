import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import imageio                          # only neccessary for gif creation


class SequentialBayes:
    """
    Sequential Bayesian learning class.
    """
    def __init__(self):
        """
        In the constructor we define our prior - zero-mean isotropic Gaussian governed by single precision parameter alpha = 2;
        N(w|0, alpha**(-1)*I)
        """
        self.a_0 = -0.3         # First parameter of the linear function.
        self.a_1 = 0.5          # Second parameter of the linear function.
        self.alpha = 2          # Precision of the prior.
        self.beta = 25          # Precision of the noise.
        self.iteration = 0      # Hold information about the current iteration.

        self.prior_mean = [0, 0]
        self.prior_cov = 1/self.alpha * np.eye(2)
        self.prior_distribution = tfd.MultivariateNormalFullCovariance(loc=self.prior_mean, covariance_matrix=self.prior_cov)

    def linear_function(self, X, noisy=True):
        """
        Target, linear function y(x,a_0,a_1) = a_0 + a_1 * x.
        By default, generated samples are also affected by Gaussian noise modeled by parameter beta.

        :param X: tf.Tensor of shape (N,), dtype=float32. Those are inputs to the linear function.
        :param noisy: boolean. Decides whether we should compute noisy or noise-free output.
        :return: tf.Tensor of shape=(N,), dtype=float32. Those are outputs from the linear function.
        """
        if noisy:
            noise_distribution = tfd.Normal(loc=0, scale=1 / np.sqrt(self.beta))
            return self.a_0 + self.a_1 * X + tf.cast(noise_distribution.sample(len(X)), tf.float32)
        else:
            return self.a_0 + self.a_1 * X

    def get_design_matrix(self, X):
        """
        Computes the design matrix of size (NxM) for feature vector X.
        Here particularly, the function simply adds the phi_0 dummy basis (equal to 1 for all elements).
        :param X: tf.Tensor of shape (N,), dtype=float32. Those are inputs to the linear function.
        :return: NxM design matrix.
        """
        N = len(X)
        M = len(self.prior_mean)
        design_mtx = np.ones((N, M))
        design_mtx[:, 1] = X
        return design_mtx

    def update_prior(self, X, T):
        """
        Single learning iteration, where we use Bayes' Theorem to calculate the new posterior over model's parameters.
        Finally, the computed posterior becomes the new prior.
        :param X: tf.Tensor of shape (N,), dtype=float32. Feature vector.
        :param T: tf.Tensor of shape=(N,), dtype=float32. Regression target.
        """
        design_mtx = self.get_design_matrix(X)

        self.posterior_cov = np.linalg.inv(np.linalg.inv(self.prior_cov) + self.beta * design_mtx.T.dot(design_mtx))
        self.posterior_mean = self.posterior_cov.dot(np.linalg.inv(self.prior_cov).dot(self.prior_mean)
                                                     + self.beta * design_mtx.T.dot(T))
        self.posterior_distribution = tfd.MultivariateNormalFullCovariance(loc=self.posterior_mean,
                                                                 covariance_matrix=self.prior_cov)
        self.prior_mean = self.posterior_mean
        self.prior_cov = self.posterior_cov
        self.prior_distribution = self.posterior_distribution

        self.iteration += 1

    def plot_prior(self):
        """
        Plot prior (posterior) distribution in parameter space. Also include the point, which indicates target parameters.
        """
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        w_0, w_1 = np.meshgrid(x, y)

        z = self.prior_distribution.prob(np.dstack((w_0, w_1)))

        plt.contourf(x, y, z, cmap='plasma')
        plt.plot(self.a_0, self.a_1, marker = 'x', c = 'orange')    # indicate target parameters in the plot
        plt.title("Prior/Posterior Plot (iteration {})".format(self.iteration))
        plt.xlabel("$w_0$")
        plt.ylabel("$w_1$")
        ax = plt.axes()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plt.savefig('Prior_Posterior-{}.png'.format(self.iteration))
        plt.clf()

    def plot_likelihood(self, X, T):
        """
        Plot likelihood distribution in parameter space. Also include the point, which indicates target parameters.
        :param X: tf.Tensor of shape (N,), dtype=float32. Feature vector.
        :param T: tf.Tensor of shape=(N,), dtype=float32. Regression target.
        """

        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        w_0, w_1 = np.meshgrid(x, y)

        least_squares_sum = 0
        for point, target in zip(X, T):
            least_squares_sum += (target - (w_0 + w_1 * point))**2
        z = np.exp(-self.beta*least_squares_sum)

        plt.contourf(x, y, z, cmap='plasma')
        plt.plot(self.a_0, self.a_1, marker='x', c='orange')  # indicate target parameters in the plot
        plt.title("Likelihood Plot (iteration {})".format(self.iteration))
        plt.xlabel("$w_0$")
        plt.ylabel("$w_1$")
        ax = plt.axes()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plt.savefig('Likelihood-{}.png'.format(self.iteration))
        plt.clf()

    def plot_data_space(self, X, T, stdevs = 1):
        """
        Plot sampled datapoints, confidence bounds, mean prediction and target function on one graph.
        :param X: tf.Tensor of shape (N,), dtype=float32. Feature vector.
        :param T: tf.Tensor of shape=(N,), dtype=float32. Regression target.
        :param stdevs: int, how large should our confidence bound be in terms of standard deviation
        """

        x = np.linspace(-1, 1, 100)
        predictions = self.prediction_mean_std(x)
        prediction_means = [x[0] for x in predictions]
        y_upper = [x[0] + stdevs * x[1] for x in predictions]
        y_lower = [x[0] - stdevs * x[1] for x in predictions]

        plt.title('Data Space (iteration {})'.format(self.iteration))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        ax = plt.axes()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # plot generated data points
        for point, target in zip(X, T):
            plt.scatter(x=point.numpy(), y=target.numpy(), marker ='o', c='blue', alpha=0.7)
        # plot confidence bounds
        plt.fill_between(x, y_upper, y_lower, where=y_upper >= y_lower, facecolor='orange', alpha=0.3)
        # plot prediction mean
        plt.plot(x, prediction_means, '-r', label='Prediction mean', c='orange', linewidth=2.0, alpha=0.8)
        # plot real function
        plt.plot(x, self.linear_function(x, noisy = False), '-r', label='Target function', c='red', linewidth=2.0, alpha=0.8)
        plt.legend(loc='upper left')
        plt.savefig('Data_Space-{}.png'.format(self.iteration))
        plt.clf()

    def prediction_mean_std(self, X):
        """
        For every sample compute mean of the corresponding Gaussian predictive distribution,
        as well as the standard deviation.
        :param X: tf.Tensor of shape (N,), dtype=float32. Feature vector.
        :return: list of tuples, where every tuple contains floats (mean, std)
        """
        no_samples = len(X)
        design_mtx = self.get_design_matrix(X)
        prediction = []
        for index in range(no_samples):
            x = design_mtx[index, :]
            predictive_std = np.sqrt(1/self.beta + x.T.dot(self.prior_cov.dot(x)))
            predictive_mean = np.array(self.prior_mean).dot(x)
            prediction.append((predictive_mean, predictive_std))
        return prediction


def run_sequential_bayes(create_gif=True):
    samples_in_batch = 1        # batch size
    no_iterations = 20          # no of learning sequences
    samples_precision = 1000    # decimal precision of a sample

    sequential_bayes = SequentialBayes()
    samples_generator = tfd.Uniform(low=-samples_precision, high=samples_precision)

    for i in range(no_iterations):
        X = samples_generator.sample(samples_in_batch) / samples_precision
        T = sequential_bayes.linear_function(X)

        sequential_bayes.plot_likelihood(X, T)
        sequential_bayes.plot_prior()
        sequential_bayes.plot_data_space(X, T)
        sequential_bayes.update_prior(X, T)

    if create_gif:
        gif_types = ['Data_Space', 'Likelihood', 'Prior_Posterior']
        for gif_type in gif_types:
            image_names = [gif_type + '-{}.png'.format(i) for i in range(no_iterations)]
            images = []
            for i in image_names:
                images.append(imageio.imread(i))
            imageio.mimsave('{}.gif'.format(gif_type), images, duration =0.3)


if __name__ == "__main__":
    tfd = tfp.distributions
    run_sequential_bayes()







