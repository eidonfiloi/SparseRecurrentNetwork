import midi
import pickle

__author__ = 'ptoth'


if __name__ == '__main__':

    midi_file_path = "../data_prepared/bach_847.mid"
    chorales_path = "../resources/jsb_chorales/JSB_Chorales.pickle"
    chorales_midi_path = "../resources/jsb_chorales/JSB_Chorales/train/2.mid"

    with open(chorales_path, 'rb') as fi:
        chorales = pickle.load(fi)

    midi_chorales = midi.read_midifile(chorales_midi_path)

    midi_file = midi.read_midifile(midi_file_path)

    x = midi_file[0]

    x_0 = x[40]
    print midi_file
