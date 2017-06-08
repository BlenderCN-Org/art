import os
import sys
import time
import json

import numpy as np
import random as rd

from midi import open_midi


# Parametrization?

# 0000002034560000
# 0000010100100001

# Generate area of used keys: key_area
# Generate area of left and right hand: left_area, right_area
# Same pattern as before? Distance?
#   Random -> Use spare fingers / Move hand
# Sustractive patter?
#   Release finger
# Aditive patter?
#   Use spare finger
# New pattern
#   Pressed keys are adove finger
#     Don't move
#   Divide key area for 1, 2 or inverted hands
#     Move hands to the areas
#     Snap fingers to closest pressed keys




def generate_specimens(specimens_count, frames):
    # 1 + 1 + 5 + 5 = 12
    specimens = np.empty((specimens_count, frames.shape[0], 12), dtype=int)
    for specimen_index in range(0, specimens.shape[0]):
        for frame_index in range(0, specimens.shape[1]):
            keyboard = []
            pattern = []
            some_pressed = False
            for key_index in range(0, specimens.shape[2]):
                keyboard.append(0)
                if frame[frame_index][key_index] == 1:
                    keyboard[-1] = 1
                    some_pressed = True
                if some_pressed:
                    pattern.append(frame[frame_index][key_index])
            



    return specimens


# EVALUATE FITNESS
# Distance traveled by finger
# Distance between fingers
# Velocity on each "frame" by finger
# Mixing fingers
def fitness_eval(specimens):
    scores = np.empty(specimens.shape[0], dtype=int)
    for specimen_index in range(0, specimens.shape[0]):
        scores[specimen_index] = np.sum(finger_distances)+left_hand_distance+right_hand_distance+mixing_score
    return scores


# Select fitests
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    if rng == 0.0:
        return rawpoints
    return high - (((high - low) * (maxs - rawpoints)) / rng)


def load_json(path):
    if not os.path.exists(path):
        return False
    else:
        start = time.time()
        while is_locked(path):
            if start - time.time() > 1:
                # Print every 1 second
                print ("!! File {} locked, please unlock".format(path))
                start = time.time()
        with open(path) as data_file:
            try:
                data = json.load(data_file)
                return data
            except:
                raise
                return False


def is_locked(path):
    if os.path.exists("{}.lock".format(path)):
        return True
    else:
        return False


def lock_file(path):
    locked_filepath = "{}.lock".format(path)
    if not os.path.exists(locked_filepath):
        open(locked_filepath, 'a').close()


def unlock_file(path):
    locked_filepath = "{}.lock".format(path)
    if os.path.exists(locked_filepath):
        os.remove(locked_filepath)


## MAIN CODE

do_help = False
if len(sys.argv) == 1:
    do_help = True
else:
    try:
        sufix = int(sys.argv[1])
        specimens_count = int(sys.argv[2])
        percent_to_clone = float(sys.argv[3])
        percent_to_select = float(sys.argv[4])
        mutation_probability = float(sys.argv[5])
    except:
        do_help = True
if do_help:
    print ("Usage:")
    print ("key2finger_evolutionary.py instance_number specimens_count percent_to_clone percent_to_select mutation_probability")
    print ("Example: key2finger_evolutionary.py 1 50 0.5 0.2 0.05")
    sys.exit()

print ("Loading instance {:d}".format(sufix))

#specimens_count = 50
#percent_to_clone = 0.5
#percent_to_select = 0.2
#mutation_probability = 0.05

filepath_base = "best_speciment_{}{}.{}"
html_filepath = filepath_base.format(sufix, "", "html")
json_filepath = filepath_base.format(sufix, "", "json")
first_save = True

# IMPORT DATA
frames = open_midi('MIDI/Hummelflug.midigram', -1)
print ("Frames: ", frames.shape[0])

# ENCODE
# Leave only frames with Key transitions
# (Delete frame when equal to frame-1).
n = 1
while n < frames.shape[0]:
    if np.array_equal(frames[n], frames[n-1]):
        frames = np.delete(frames, (n), axis=0)
    else:
        n += 1
print ("Reduced frames:", frames.shape[0])

# Smooth:
# If a key is released and presed again really quicly
# remove that transition.

# FIRST GENERATION
print ("Generating Specimens")
start = time.time()
specimens = generate_specimens(specimens_count, frames)
end = time.time()
elapsed = end - start
print ("Specimens Generated in {:.2f} seconds".format(elapsed))

# Load previous specimen
print ("Loading last specimen")
backup_filepath = filepath_base.format(sufix, "_backup", "json")
if os.path.exists(backup_filepath):
    best_saved_specimen = load_json(backup_filepath)
    specimens[0] = best_saved_specimen
    print ("Backup loaded")
else:
    print ("Backup not found")

while 1:
    print (".")
    print ("Fitness Evaluation")
    start = time.time()
    scores = 50000-fitness_eval(specimens) # Invert scores. Value to maximize
    end = time.time()
    elapsed = end - start
    print ("Fitness Evaluated in {:.2f} seconds: {:d}".format(elapsed, scores.max()))
    best_specimen = np.argmax(scores)
    print ("Saving speciment {}".format(best_specimen))
    # Specimen to Text
    specimen_text = """<style>
body {
    background: black;
}

.finger_01 {
    color: red;
}

.finger_02 {
    color: red;
}

.finger_03 {
    color: red;
}

.finger_04 {
    color: red;
}

.finger_05 {
    color: red;
}

.finger_06 {
    color: green;
}

.finger_07 {
    color: green;
}

.finger_08 {
    color: green;
}

.finger_09 {
    color: green;
}

.finger_10 {
    color: green;
}
</style>
<p>"""
    for frame_index in range(0, specimens.shape[1]):
        for finger_index in range(0, specimens.shape[2]):
            if specimens[best_specimen][frame_index][finger_index] > 0:
                specimen_text = "{0}<span class=\"finger_{1:02d}\">{1:02d}</span>".format(specimen_text, specimens[best_specimen][frame_index][finger_index])
            else:
                specimen_text = "{}{:02d}".format(specimen_text, specimens[best_specimen][frame_index][finger_index])
        specimen_text = "{}{}".format(specimen_text, "<br>\n")
    specimen_text = "{}{}".format(specimen_text, "</p>")

    if first_save:
        # Security check
        if os.path.exists(html_filepath):
            print ("Error saving {}, file already exists".format(html_filepath))
            sys.exit()
        if os.path.exists(json_filepath):
            print ("Error saving {}, file already exists".format(html_filepath))
            sys.exit()
        first_save = False
    lock_file(html_filepath)
    with open(html_filepath, "w") as text_file:
        text_file.write(specimen_text)
    unlock_file(html_filepath)
    lock_file(json_filepath)
    with open(json_filepath, 'w', encoding='utf-8') as anim_file:
        json.dump(specimens[best_specimen].tolist(), anim_file, sort_keys=True, indent=4, separators=(',', ': '))
    unlock_file(json_filepath)

    scores = scale_linear_bycolumn(scores, 1.0, 0.0)
    softmax_scores = softmax(scores)

    fitness_selection = np.random.choice(np.arange(specimens.shape[0]), int(specimens_count*percent_to_select), p=softmax_scores)

    print ("Crossover Specimens")
    start = time.time()
    new_specimens = np.empty((specimens_count, frames.shape[0], 12), dtype=int)
    cloned_count = int(specimens_count*percent_to_clone)
    new_count = specimens_count - cloned_count
    # Clone
    for n in range(0, cloned_count):
        new_specimens[n] = specimens[np.random.choice(fitness_selection)]
    # Always keep best specimen
    new_specimens[0] = specimens[best_specimen]
    # Load specimens from other instances
    for instance_number in range(0, 8):
        if instance_number == sufix:
            continue
        saved_filepath = filepath_base.format(instance_number, "", "json")
        if os.path.exists(saved_filepath):
            best_saved_specimen = load_json(saved_filepath)
            new_specimens[1+instance_number] = best_saved_specimen
            print ("Loaded instance {} specimen".format(instance_number))
    # Crossover
    for n in range(cloned_count, specimens_count):
        mixer_size = (specimens.shape[1], specimens.shape[2])
        mixer = np.random.random_integers(low=0, high=1, size=mixer_size)
        for nn in range(0, mixer.shape[0]):
            if mixer[nn][0] == 1:
                mixer[nn] = np.ones(mixer.shape[1], dtype=int)
            else:
                mixer[nn] = np.zeros(mixer.shape[1], dtype=int)
        specimen_a = specimens[np.random.choice(fitness_selection)]
        specimen_b = specimens[np.random.choice(fitness_selection)]

        new_specimens[n] = (specimen_a*mixer)+(specimen_b*(1-mixer))
    end = time.time()
    elapsed = end - start
    print ("Specimens Crossover in {:.2f} seconds".format(elapsed))

    # Mutate
    print ("Mutate Frames")
    start = time.time()
    mutations = generate_specimens(specimens_count, frames)
    mutation_count = 0
    total_count = 0
    for mutations_index in range(cloned_count, specimens_count):
        if not mutation_probability > rd.random():
            continue
        total_count += 1
        if 0.5 > rd.random():
            # Half of the time mutate several Frames
            for frame_index in range(0, new_specimens.shape[1]):
                if mutation_probability > rd.random():
                    mutation_count += 1
                    new_specimens[mutations_index][frame_index] = mutations[mutations_index][frame_index]
        else:
            # Half of the time mutate only one Frame
            mutation_count += 1
            random_frame = rd.randint(0, mutations.shape[1]-1)
            new_specimens[mutations_index][random_frame] = mutations[mutations_index][random_frame]
    end = time.time()
    elapsed = end - start
    print ("{} Frames Mutated on {} Specimens in {:.2f} seconds".format(mutation_count, total_count, elapsed))

    specimens = new_specimens
