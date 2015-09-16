__author__ = 'eidonfiloi'

from data_io.audio_data_utils import *


if __name__ == "__main__":

    #sample frequency in Hz
    freq = 44100
    #length of clips for training. Defined in seconds
    clip_len = 10
    #block sizes used for training - this defines the size of our input state
    block_size = freq / 10
    #Used later for zero-padding song sequences
    max_seq_len = int(round((freq * clip_len) / block_size))
    #Step 1 - convert MP3s to WAVs
    new_directory = convert_folder_to_wav('resources/bach_goldberg/', freq)
    #Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
    convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, 'data_prepared/bach_goldberg_aria_10', useTimeDomain=False)

