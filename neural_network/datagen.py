import numpy as np                    # linear algebra, vector operations, numpy arrays
import matplotlib.pyplot as plt       # allows for plotting
from scipy.io import wavfile          # data IO
import os

import config as CFG


class DataGen:

    def __init__(self):
        self.working_dir = 'numbers1/'
        self.orig_wavfiles = os.listdir(self.working_dir)

    def reduce_audio_images(self):
        """
        Converts audio samples into images and reduces them based on configuration values
        """
        for count, file in enumerate(self.orig_wavfiles):

            audio_sample = os.path.join(self.working_dir, file)

            if os.path.isfile(audio_sample):

                # convert the wavfile into its amplitude and audio channels
                amplitude, audio = wavfile.read(audio_sample)
                time = np.arange(0, len(audio)) / amplitude

                print(f'IMAGE {count} (filename {audio_sample})')
                if CFG.TEST_CODE:
                    print(f'amplitude: {amplitude}')
                    print(f'audio (shape {audio}): {audio}')

                # setup plot with reduced dimensions
                _, _ = plt.subplots(figsize=(CFG.REDUCE_DIM, CFG.REDUCE_DIM))

                # plot reduced image
                plt.plot(time, audio, 'blue', alpha=1)
                plt.ylim([-20000, 20000])
                plt.xlim([0, 2])
                plt.savefig(CFG.REDUCED_DIR + file[:-4] + 'r')

                # uncomment to view all images
                # plt.show()
