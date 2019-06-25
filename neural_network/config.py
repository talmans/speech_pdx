from os import getcwd

TEST_CODE = False    # True if testing code, False if running neural network fully

if TEST_CODE:
    print(f'You are in CODE TESTING mode. To switch to regular program mode,\n'
          f'change `TEST_CODE = True` in config.py')
    EPOCHS = 2
    TRAIN_PERCENT = .90
else:
    EPOCHS = 50
    TRAIN_PERCENT = .80

# Directories and Files
PATH = getcwd()
GRAPHS_DIR = PATH + '/accuracy_graphs/'  # directory for storing accuracy graphs
REDUCED_DIR = PATH + '/reduced_images/'  # directory for storing reduced images

# Image Sources and Processing Specs
TOMS_DATA = 'toms_graphs/'
VIRA_ENV = 'envel08/'
VIRA_SPECT = 'specs08/'
NUM1 = 'numbers1'
NUM2 = 'numbers2'
DATA_SOURCE = PATH + '/image_data/' + VIRA_SPECT     # select directory for grabbing data from
GENERATE_RESIZED_IMAGES = True      # rescale the images
SCALE_PERCENT = .6                  # scale image while maintaining aspect ratio

# Datagen Audio Conversion Specs
AUDIO_SOURCE = 'numbers1/'          # select directory with audio data
CONVERT_AUDIO = False               # if True, convert provided wavfiles over to images, otherwise, no conversion occurs
REDUCE_DIM = 2                      # figure size for audio-to-image data

# NN Hyperparameters
ETA = 0.001
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = False
DROP_VAL = 0.3
BATCH_SIZE = 28
ACTIVATE_FN = ["relu", "relu", "relu", "softmax"]

# NN Image Specs
AUGMENT_IMAGES = False   # If True, augments images, if False, uses original images
MNIST_DATA = False       # True if using mnist data, False if using custom data

ROW = 28
COL = 28
DEPTH = 1

if MNIST_DATA:
    TOTAL_LABELS = 10
else:
    TOTAL_LABELS = 13

# Statistical Graph Specs
ANNOTATE = False                            # if True, annotates values above a certain accuracy, if False, no annotation
ANNOTATE_LEVEL = 75                         # percentage at which to annotate
COLOR_ACCURACY = ['#f15d29', '#37744a']     # green and blue line
COLOR_COST = ['#f15d29', '#37744a']         # red and green lines
PLOT_COST = True                            # if True, plot cost graph, if False, cost is not plotted
FIG_WIDTH = 8                               # width for accuracy/cost figures
FIG_HEIGHT = 6                              # height for accuracy/cost figures

