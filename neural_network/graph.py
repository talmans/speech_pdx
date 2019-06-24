import numpy as np                    # linear algebra, vector operations, numpy arrays
import matplotlib.pyplot as plt       # allows for plotting
import seaborn as sns                 # accuracy line plot
from os.path import exists            # check that path exists
from os import mkdir, chdir           # directory operations

import config as CFG                  # program constants


class Graph:

    def __init__(self, experiment_count: int, total_tr_data: int, tr_accuracy: [int], total_test_data: int,
                 test_accuracy: [int], function_name: str, batch_size: int, title: str="CNN Accuracy Plot"):
        """
        Initialize the Graph object
        :param experiment_count: count for different experiments (useful for separating files later on)
        :param total_tr_data: total data size
        :param tr_accuracy: accuracy values collected from each epoch
        :param total_test_data: total data size
        :param test_accuracy: accuracy values collected from each epoch
        :param function_name: name of activation function
        :param batch_size: mini batch size
        :param experiment_count: the current count of the experiment being graphed
        :param title: a replacement title if desired, otherwise the default title will be used
        """
        self.title = title
        self.total_tr_data = total_tr_data
        self.total_test_data = total_test_data

        # accuracy and last epoch
        self.tr_accuracy = tr_accuracy.ravel()
        self.last_epoch = str(int(self.tr_accuracy[CFG.EPOCHS - 1]))
        self.last_epoch_per = self.tr_accuracy[CFG.EPOCHS - 1]
        self.tr_mean = np.mean(self.tr_accuracy[-(CFG.EPOCHS - 10):])
        self.tr_mean_str = "{:.2f}".format(self.tr_mean)

        self.test_accuracy = test_accuracy.ravel()
        self.last_epoch = str(int(self.test_accuracy[CFG.EPOCHS - 1]))
        self.last_epoch_per = self.test_accuracy[CFG.EPOCHS - 1]
        self.test_mean = np.mean(self.test_accuracy[-(CFG.EPOCHS - 10):])
        self.test_mean_str = "{:.2f}".format(self.test_mean)

        # hyperparameters and string equivalents (for formatting easily in overleaf)
        self.function_name = function_name
        self.batch_size = batch_size

        # data table specifications
        self.cell_length = len(str(" " + str(self.total_tr_data) + "/" + str(self.total_tr_data) + " "))
        self.total_cells = 3
        self.max_length = (self.cell_length + 1) * (self.total_cells)
        self.tablefile = "table.txt"
        self.experiment_count = experiment_count

    def prep_titles(self, cost_title: str="") -> (str, str):
        """
        Prepare titles for plot and saved images
        :param cost_title: if provided, a string for the cost plot title
        :return: title for the plot, img_title for the saved image
        """
        img_title = self.function_name + \
                    '_batch' + str(self.batch_size)

        if cost_title == "":
            img_title = str(self.experiment_count) + '_accuracy_plot_' + img_title
            title = self.title + \
                    '\n' + self.function_name + ", " + \
                    'mini-batch size: ' + str(self.batch_size) + \
                    '\nAvg Last 10 Epochs: Training ' + self.tr_mean_str + '%, Testing ' + self.test_mean_str + '%'
        else:
            img_title = str(self.experiment_count) + '_cost_plot_' + img_title
            title = cost_title

        print(f'\nexperiment: {img_title}')
        return title, img_title

    def plot_accuracy(self):
        """
        Plot accuracy in each epoch
        :param legend_name: name equating to line in plot
        """
        plot_title, img_title = self.prep_titles("")
        test_legend = ['training data', 'test data']

        # Data for plotting x- and y-axis
        x = np.arange(1, CFG.EPOCHS + 1)
        y = [self.tr_accuracy, self.test_accuracy]

        # prints x and y-axis values
        print(f'x: {x}')
        print(f'training: {self.tr_accuracy}')
        print(f'test: {self.test_accuracy}')

        plt.figure(figsize=(CFG.FIG_WIDTH, CFG.FIG_HEIGHT))

        # Create the lineplot
        for line in range(2):
            ax = sns.lineplot(x=x, y=y[line], color=CFG.COLOR_ACCURACY[line], label=test_legend[line])

        if CFG.ANNOTATE:
            ax.set(xlabel='Epochs',
                   ylabel='Accuracy (%)',
                   title=plot_title,
                   xlim=(1, CFG.EPOCHS + 2),
                   ylim=(0, 119))

            for line in range(2):
                for e in range(0, CFG.EPOCHS):
                    if y[line][e] > CFG.ANNOTATE_LEVEL:
                        value = "{:.2f}".format(y[line][e])
                        label = "epoch " + str(e + 1) + "\n" + value + "%"
                        plt.annotate(label,
                                     xy=(x[e], y[line][e]),
                                     alpha=1,
                                     size=9,
                                     rotation=45,
                                     textcoords='offset pixels', xytext=(0, 7),
                                     ha='left', va='bottom')
        else:
            ax.set(xlabel='Epochs',
                   ylabel='Accuracy (%)',
                   title=plot_title,
                   xlim=(1, CFG.EPOCHS),
                   ylim=(0, 102))

        ax.legend(loc='best')

        self.save_plot(img_title)

        # uncomment to show the plots
        plt.show()

    def plot_cost(self, cost_tr_data: [[int]], cost_test_data: [[int]], title: str="Cost"):
        """
        Plot cost in each epoch
        :param cost_data: array of cost values
        :param title: title for plot
        """
        plot_title, img_title = self.prep_titles(title)
        test_legend = ['training data', 'test data']

        # Data for plotting x- and y-axis
        x = np.arange(1, CFG.EPOCHS + 1)
        y = [cost_tr_data, cost_test_data]

        # prints x and y-axis values
        print(f'COST:')
        print(f'x: {x}')
        print(f'y: {y}')
        max_y1 = np.amax(y[0]) + 0.1
        max_y2 = np.amax(y[1]) + 0.1
        max_y = max(max_y1, max_y2)

        plt.figure(figsize=(CFG.FIG_WIDTH, CFG.FIG_HEIGHT))

        # Create the lineplot
        for line in range(2):
            ax = sns.lineplot(x=x, y=y[line], color=CFG.COLOR_COST[line], label=test_legend[line])

        ax.legend(loc='best')
        ax.set(xlabel='Epochs',
               ylabel='Cost',
               title=plot_title,
               xlim=(1, CFG.EPOCHS),
               ylim=(0, max_y))

        self.save_plot(img_title)
        plt.show()

    @staticmethod
    def save_plot(title: str):
        """
        Saves plot to specified directory.
        :param title: title of plot
        """
        path = CFG.GRAPHS_DIR

        # create directory if it doesn't exist
        if not exists(path):
            mkdir(path)

        # save plot to specified directory
        img = path + title + '.png'
        plt.savefig(img)

    def pretty_string(self, item: str, end: bool=False):
        """
        Pretty formats the item passed in so that the overall matrix looks pretty and is more easily readable
        :param item: item of string type
        :param end: True if there should be an ending border, False otherwise
        :return pretty formatted string
        """
        new_item = str(item)
        item_length = len(new_item)

        if item == "Functions":
            item_length += 2

        extra_spaces = self.cell_length - item_length
        left_spaces = extra_spaces // 2
        right_spaces = extra_spaces - left_spaces

        if end:
            return '|' + (left_spaces * ' ') + new_item + (right_spaces * ' ') + '|'
        return '|' + (left_spaces * ' ') + new_item + (right_spaces * ' ')

    def print_border_line(self):
        """
        Prints a border line
        :return formatted breakline string
        """
        return (self.max_length + 1) * '-' + '\n'

    def tabular_data(self):
        """
        Print tabular data for experiment including hyperparameters and resulting accuracy to file
        e.g. name, batch, accuracy
        """
        path = CFG.GRAPHS_DIR
        chdir(path)

        if self.experiment_count == 1:
            f = open(self.tablefile, 'w')
            f.write(self.print_border_line())
            f.write(self.table_header())
            f.write(self.print_border_line())
            f.write(self.pretty_string("Functions"))
            f.write(self.pretty_string("Batch Size"))
            f.write(self.pretty_string("Training (%)"))
            f.write(self.pretty_string("Testing (%)", True))
            f.write('\n')
            f.write(self.print_border_line())
            f.close()

        f = open(self.tablefile, 'a')
        f.write(self.pretty_string(self.function_name))
        f.write(self.pretty_string(str(self.batch_size)))
        f.write(self.pretty_string(self.tr_mean_str))
        f.write(self.pretty_string(self.test_mean_str, True))
        f.write('\n')
        f.close()

    def table_header(self):
        """
        Prints header for table data
        """
        title = 'HYPERPARAMETER FINE-TUNING RESULTS'
        title_len = len(title)
        extra_spaces = self.max_length - title_len
        left_spaces = extra_spaces // 2
        right_spaces = extra_spaces - left_spaces - 1

        return '|  ' + (left_spaces * ' ') + title + (right_spaces * ' ') + '  |\n'

