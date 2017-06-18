import os

from piano import specimen

from midi import open_midi


## MAIN CODE
try:
    # Attempts to load configuration
    from config import Config
except:
    print ("Missing config.py")
    sys.exit()

midigram_filepath = Config.midigram_filepath
octave_shift = Config.octave_shift
frame_steps = Config.frame_steps # Window
max_steps = Config.max_steps # Quality
sufix = Config.sufix
specimens_count = Config.specimens_count
percent_to_clone = Config.percent_to_clone
percent_to_select = Config.percent_to_select
mutation_probability = Config.mutation_probability

if not os.path.exists(midigram_filepath):
    print ("Midigram file {} does not exists.".format(midigram_filepath))
    sys.exit()

# IMPORT DATA
full_frames, reduced_frames, condensed_frame_index = open_midi(midigram_filepath, octave_shift)
print ("Full Frames: ", full_frames.shape[0])
print ("Reduced Frames: ", reduced_frames.shape[0])


specimen_zero = specimen(10)
