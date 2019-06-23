from os import getcwd

TEST_CODE = False    # True if testing code, False if running neural network fully

if TEST_CODE:
    print(f'You are in CODE TESTING mode. To switch to regular program mode,\n'
          f'change `TEST_CODE = True` in config.py')
    EPOCHS = 2
    TRAIN_PERCENT = .90
else:
    EPOCHS = 60
    TRAIN_PERCENT = .80

# Hyperparameters
ETA = 0.0009
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = False
DROP_VAL = 0.3

# Training/Testing Specs
BATCH_SIZE = 28
ACTIVATE_FN = ["relu", "relu", "softmax"]
AUGMENT_IMAGES = False   # If True, augments images, if False, uses original images

# NN Image Specs
MNIST_DATA = False   # True if using mnist data, False if using custom data

if MNIST_DATA:
    ROW = 28
    COL = 28
    DEPTH = 1
    TOTAL_LABELS = 10
else:
    ROW = 28
    COL = 28
    DEPTH = 3
    TOTAL_LABELS = 13

# DataGen Image Specs
GENERATE_REDUCED_IMAGES = False
REDUCE_DIM = .28                            # .28 selected to work as replacement to MNIST dataset functions, settings, etc

# Graph Specs
ANNOTATE = False                            # if True, annotates values above a certain accuracy, if False, no annotation
ANNOTATE_LEVEL = 75                         # percentage at which to annotate
COLOR_ACCURACY = ['#f15d29', '#37744a']     # green and blue line
COLOR_COST = ['#f15d29', '#37744a']         # red and green lines
PLOT_COST = True                            # if True, plot cost graph, if False, cost is not plotted

# Directories and Files
PATH = getcwd()
GRAPHS_DIR = PATH + '/accuracy_graphs/'  # directory for storing accuracy graphs
REDUCED_DIR = PATH + '/reduced_images/'  # directory for storing reduced images
