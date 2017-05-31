import time
import json

import numpy as np
import random as rd

from midi import open_midi


def generate_specimens(specimens_count, frames):
    specimens = np.empty((specimens_count, frames.shape[0], 88), dtype=int)

    for specimen_index in range(0, specimens.shape[0]):
        for frame_index in range(0, specimens.shape[1]):
            key_pressed_count = np.count_nonzero([frames[frame_index]>0])
            if key_pressed_count == 0:
                continue
            elif key_pressed_count == 1:
                finger_distribution = np.empty(10, dtype=int)
                finger_distribution.fill(rd.randint(1, 10))
            elif key_pressed_count > 1:
                if 0.05 > rd.random():
                    finger_distribution = np.random.permutation(10)+1
                else:
                    finger_distribution = np.roll(np.arange(10)+1, rd.randint(0, 10))
            specimens[specimen_index][frame_index] = np.array(frames[frame_index])
            nf=0
            it = np.nditer(specimens[specimen_index][frame_index], flags=['f_index'])
            while not it.finished:
                if specimens[specimen_index][frame_index][it.index]>0:
                    specimens[specimen_index][frame_index][it.index] = finger_distribution[nf]
                    nf+=1
                    if nf >= key_pressed_count:
                        break
                it.iternext()
    return specimens


# EVALUATE FITNESS
# Distance traveled by finger
# Distance between fingers
# Velocity on each "frame" by finger
# Mixing fingers
def fitness_eval(specimens):
    mixing_cost = 2
    pinky_min = 6  # Left
    pinky_max = 2  # Right
    ring_min = 4   # Left
    ring_max = 1   # Right
    index_min = 6  # Right
    index_max = 1  # Left
    thumb_min = 10 # Right
    thumb_max = -2 #4  # Left

    scores = np.empty(specimens.shape[0], dtype=int)
    for specimen_index in range(0, specimens.shape[0]):
        finger_distances = np.zeros(10, dtype=int) # Initial finger distance
        finger_previous_position = np.arange(44, 54, dtype=int) # Initial finger positions
        mixing_score = 0 # Initial mixing score
        for frame_index in range(0, specimens.shape[1]):
            # Get pressed keys
            keys_with_finger = [specimens[specimen_index][frame_index]>0]
            for finger_index in specimens[specimen_index][frame_index][keys_with_finger]:
                # TODO the following is always true!
                key_with_finger = [specimens[specimen_index][frame_index]==finger_index] # True for finger_index
                #if not np.any(key_with_finger):
                #    continue
                current_position = np.argmax(key_with_finger) # What index is True?
                # TODO index 3403658297748896047 is out of bounds for axis 0 with size 10
                #            9223372036854775807
                try:
                    dist = finger_previous_position[finger_index-1]-current_position
                except:
                    print (finger_index, current_position)
                    raise
                if dist < 0:
                    dist = dist *-1
                try:
                    finger_distances[finger_index-1] += dist
                except:
                    print (finger_index)
                    raise
                finger_previous_position[finger_index-1] = current_position
            finger_position = finger_previous_position
            # Check finger mixing
            keys_with_finger = [specimens[specimen_index][frame_index]>0]
            if np.any(keys_with_finger):
                key_a_iter = np.nditer(specimens[specimen_index][frame_index][keys_with_finger], flags=['f_index'])
                while not key_a_iter.finished:
                    key_b_iter = np.nditer(specimens[specimen_index][frame_index][keys_with_finger], flags=['f_index'])
                    while not key_b_iter.finished:
                        if key_a_iter[0] < key_b_iter[0] and finger_position[key_a_iter[0]-1] > finger_position[key_b_iter[0]-1]:
                            # If a finger with small index
                            # is in a higher position
                            mixing_score += mixing_cost
                        if key_b_iter[0] > key_b_iter[0] and finger_position[key_a_iter[0]-1] < finger_position[key_b_iter[0]-1]:
                            # If a finger with large index
                            # is in a lower position
                            mixing_score += mixing_cost
                        key_b_iter.iternext()
                    key_a_iter.iternext()

        scores[specimen_index] = np.sum(finger_distances)+mixing_score
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



## MAIN CODE

specimens_count = 50
percent_to_clone = 0.5
percent_to_select = 0.2
mutation_probability = 0.05

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
    with open("best_speciment.html", "w") as text_file:
        text_file.write(specimen_text)
    with open('best_speciment.json', 'w', encoding='utf-8') as anim_file:
        json.dump(specimens[best_specimen].tolist(), anim_file, sort_keys=True, indent=4, separators=(',', ': '))

    scores = scale_linear_bycolumn(scores, 1.0, 0.0)
    softmax_scores = softmax(scores)

    fitness_selection = np.random.choice(np.arange(specimens.shape[0]), int(specimens_count*percent_to_select), p=softmax_scores)

    print ("Crossover Specimens")
    start = time.time()
    new_specimens = np.empty((specimens_count, frames.shape[0], 88), dtype=int)
    cloned_count = int(specimens_count*percent_to_clone)
    new_count = specimens_count - cloned_count
    # Clone
    for n in range(0, cloned_count):
        new_specimens[n] = specimens[np.random.choice(fitness_selection)]
    # Always keep best specimen
    new_specimens[0] = specimens[best_specimen]
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
    for mutations_index in range(0, new_specimens.shape[0]):
        if mutation_probability < rd.random():
            continue
        total_count += 1
        for frame_index in range(0, new_specimens.shape[1]):
            if mutation_probability > rd.random():
                mutation_count += 1
                #mutations = generate_specimens(1, frames)
                #new_specimens[mutations_index][frame_index] = mutations[0][frame_index]
                new_specimens[mutations_index][frame_index] = mutations[mutations_index][frame_index]
    end = time.time()
    elapsed = end - start
    print ("{} Frames Mutated on {} Specimens in {:.2f} seconds".format(mutation_count, total_count, elapsed))

    specimens = new_specimens
