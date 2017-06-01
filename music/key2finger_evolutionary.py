import os
import sys
import time
import json

import numpy as np
import random as rd

from midi import open_midi


def generate_specimens(specimens_count, frames):
    specimens = np.empty((specimens_count, frames.shape[0], 88), dtype=int)

    for specimen_index in range(0, specimens.shape[0]):
        for frame_index in range(0, specimens.shape[1]):
            mixing_detected = True
            imposible_distance = True
            try_count = 0
            while mixing_detected or imposible_distance:
                try_count += 1
                if try_count > 100000:
                    print ("Too many attempts to generate finger position on frame {}".format(frame_index))
                    print ("Last status:")
                    print (current_frame_finger_positions)
                    sys.exit()
                key_pressed_count = np.count_nonzero([frames[frame_index]>0])
                if key_pressed_count == 0:
                    specimens[specimen_index][frame_index] = np.zeros(specimens.shape[2], dtype=int)
                    # continue
                    mixing_detected = False
                    break
                elif key_pressed_count == 1:
                    finger_distribution = np.empty(10, dtype=int)
                    finger_distribution.fill(rd.randint(1, 10))
                elif key_pressed_count > 1:
                    finger_distribution = np.zeros(key_pressed_count, dtype=int)
                    last_random = 0
                    for fd in range(0, finger_distribution.shape[0]):
                        last_random = rd.randint(last_random+1, 10-(key_pressed_count-fd)+1)
                        finger_distribution[fd] = last_random
                    finger_distribution = np.roll(finger_distribution, rd.randint(0, key_pressed_count))
                specimens[specimen_index][frame_index] = np.array(frames[frame_index])
                current_frame_finger_positions = np.zeros(10, dtype=int)
                nf=0
                it = np.nditer(specimens[specimen_index][frame_index], flags=['f_index'])
                while not it.finished:
                    if specimens[specimen_index][frame_index][it.index]>0:
                        specimens[specimen_index][frame_index][it.index] = finger_distribution[nf]
                        current_frame_finger_positions[finger_distribution[nf]-1] = it.index+1
                        nf+=1
                        if nf >= key_pressed_count:
                            break # Stop asigning fingers
                    it.iternext()
                # Check imposible distances
                imposible_distance = False
                pinky_anular = 4
                pinky_middle = 8
                pinky_index  = 12
                pinky_thumb  = 18
                #
                anular_middle = 4
                anular_index  = 8
                anular_thumb  = 14
                #
                middle_index  = 4
                middle_thumb  = 10
                #
                index_thumb  = 6

                if key_pressed_count > 1:
                    # Pinky Left
                    if current_frame_finger_positions[0] > 0:
                        # Pinky Anular Left
                        if current_frame_finger_positions[1] > 0:
                            if current_frame_finger_positions[0] < current_frame_finger_positions[1]-pinky_anular:
                                imposible_distance = True
                                continue
                        # Pinky Middle Left
                        if current_frame_finger_positions[2] > 0:
                            if current_frame_finger_positions[0] < current_frame_finger_positions[2]-pinky_middle:
                                imposible_distance = True
                                continue
                        # Pinky Index Left
                        if current_frame_finger_positions[3] > 0:
                            if current_frame_finger_positions[0] < current_frame_finger_positions[3]-pinky_index:
                                imposible_distance = True
                                continue
                        # Pinky Thumb Left
                        if current_frame_finger_positions[4] > 0:
                            if current_frame_finger_positions[0] < current_frame_finger_positions[4]-pinky_thumb:
                                imposible_distance = True
                                continue
                    # Anular Left
                    if current_frame_finger_positions[1] > 0:
                        # Anular Middle Left
                        if current_frame_finger_positions[2] > 0:
                            if current_frame_finger_positions[1] < current_frame_finger_positions[2]-anular_middle:
                                imposible_distance = True
                                continue
                        # Anular Index Left
                        if current_frame_finger_positions[3] > 0:
                            if current_frame_finger_positions[1] < current_frame_finger_positions[3]-anular_index:
                                imposible_distance = True
                                continue
                        # Anular Thumb Left
                        if current_frame_finger_positions[4] > 0:
                            if current_frame_finger_positions[1] < current_frame_finger_positions[4]-anular_thumb:
                                imposible_distance = True
                                continue
                    # Middle Left
                    if current_frame_finger_positions[2] > 0:
                        # Middle Index Left
                        if current_frame_finger_positions[3] > 0:
                            if current_frame_finger_positions[2] < current_frame_finger_positions[3]-middle_index:
                                imposible_distance = True
                                continue
                        # Middle Thumb Left
                        if current_frame_finger_positions[4] > 0:
                            if current_frame_finger_positions[2] < current_frame_finger_positions[4]-middle_thumb:
                                imposible_distance = True
                                continue
                    # Index Left
                    if current_frame_finger_positions[3] > 0:
                        # Index Thumb Left
                        if current_frame_finger_positions[4] > 0:
                            if current_frame_finger_positions[3] < current_frame_finger_positions[4]-index_thumb:
                                imposible_distance = True
                                continue

                    # Pinky Right
                    if current_frame_finger_positions[5] > 0:
                        # Pinky Anular Right
                        if current_frame_finger_positions[6] > 0:
                            if current_frame_finger_positions[5] < current_frame_finger_positions[6]-pinky_anular:
                                imposible_distance = True
                                continue
                        # Pinky Middle Right
                        if current_frame_finger_positions[7] > 0:
                            if current_frame_finger_positions[5] < current_frame_finger_positions[7]-pinky_middle:
                                imposible_distance = True
                                continue
                        # Pinky Index Right
                        if current_frame_finger_positions[8] > 0:
                            if current_frame_finger_positions[5] < current_frame_finger_positions[8]-pinky_index:
                                imposible_distance = True
                                continue
                        # Pinky Thumb Right
                        if current_frame_finger_positions[9] > 0:
                            if current_frame_finger_positions[5] < current_frame_finger_positions[9]-pinky_thumb:
                                imposible_distance = True
                                continue
                    # Anular Right
                    if current_frame_finger_positions[6] > 0:
                        # Anular Middle Right
                        if current_frame_finger_positions[7] > 0:
                            if current_frame_finger_positions[6] < current_frame_finger_positions[7]-anular_middle:
                                imposible_distance = True
                                continue
                        # Anular Index Right
                        if current_frame_finger_positions[8] > 0:
                            if current_frame_finger_positions[6] < current_frame_finger_positions[8]-anular_index:
                                imposible_distance = True
                                continue
                        # Anular Thumb Right
                        if current_frame_finger_positions[9] > 0:
                            if current_frame_finger_positions[6] < current_frame_finger_positions[9]-anular_thumb:
                                imposible_distance = True
                                continue
                    # Middle Right
                    if current_frame_finger_positions[7] > 0:
                        # Middle Index Right
                        if current_frame_finger_positions[8] > 0:
                            if current_frame_finger_positions[7] < current_frame_finger_positions[8]-middle_index:
                                imposible_distance = True
                                continue
                        # Middle Thumb Right
                        if current_frame_finger_positions[9] > 0:
                            if current_frame_finger_positions[7] < current_frame_finger_positions[9]-middle_thumb:
                                imposible_distance = True
                                continue
                    # Index Right
                    if current_frame_finger_positions[8] > 0:
                        # Index Thumb Right
                        if current_frame_finger_positions[9] > 0:
                            if current_frame_finger_positions[8] < current_frame_finger_positions[9]-index_thumb:
                                imposible_distance = True
                                continue

                # Check mixing
                jump_count = 0
                current_hand = None
                previous_hand = None
                for n_key in range(0, specimens.shape[2]):
                    if specimens[specimen_index][frame_index][n_key] > 0 and specimens[specimen_index][frame_index][n_key] < 6:
                        current_hand = 0
                    elif specimens[specimen_index][frame_index][n_key] > 5:
                        current_hand = 1
                    if previous_hand is None:
                        previous_hand = current_hand
                    if previous_hand != current_hand:
                        jump_count += 1
                        previous_hand = current_hand
                    if jump_count > 1:
                        mixing_detected = True
                        break # Try new iteration
                if jump_count < 2:
                    mixing_detected = False

                # Check mixing left hand
                for fi in range(0, 5):
                    for fii in range(fi, 5):
                        if current_frame_finger_positions[fii] == 0 or current_frame_finger_positions[fi] == 0:
                            continue
                        if current_frame_finger_positions[fii] < current_frame_finger_positions[fi]:
                            mixing_detected = True
                            break
                    if mixing_detected:
                        break
                for fi in range(5, 10):
                    for fii in range(fi, 10):
                        if current_frame_finger_positions[fii] == 0 or current_frame_finger_positions[fi] == 0:
                            continue
                        if current_frame_finger_positions[fii] < current_frame_finger_positions[fi]:
                            mixing_detected = True
                            break
                    if mixing_detected:
                        break

    return specimens


# EVALUATE FITNESS
# Distance traveled by finger
# Distance between fingers
# Velocity on each "frame" by finger
# Mixing fingers
def fitness_eval(specimens):
    mixing_cost = 2
    energy_cost_pinky = 3
    energy_cost_anular = 2
    energy_cost_middle = 1
    energy_cost_index = 0
    energy_cost_thumb = 0
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
        current_frame_finger_positions = np.zeros([specimens.shape[1], 10], dtype=int)
        finger_previous_position = np.arange(44, 54, dtype=int) # Initial finger positions
        mixing_score = 0 # Initial mixing score
        previous_frame_left_hand_position = 42
        previous_frame_right_hand_position = 47
        left_hand_distance = 0
        right_hand_distance = 0
        for frame_index in range(0, specimens.shape[1]):
            # Get pressed keys
            keys_with_finger = [specimens[specimen_index][frame_index]>0]
            for finger_index in specimens[specimen_index][frame_index][keys_with_finger]:
                key_with_finger = [specimens[specimen_index][frame_index]==finger_index] # True for finger_index
                current_position = np.argmax(key_with_finger) # What index is True?
                current_frame_finger_positions[frame_index][finger_index-1] = current_position+1
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

            # Check energy eficiency
            if finger_position[0] > 0:
                mixing_score += energy_cost_pinky
            if finger_position[1] > 0:
                mixing_score += energy_cost_anular
            if finger_position[2] > 0:
                mixing_score += energy_cost_middle
            if finger_position[3] > 0:
                mixing_score += energy_cost_index
            if finger_position[4] > 0:
                mixing_score += energy_cost_thumb
            #
            if finger_position[9] > 0:
                mixing_score += energy_cost_pinky
            if finger_position[8] > 0:
                mixing_score += energy_cost_anular
            if finger_position[7] > 0:
                mixing_score += energy_cost_middle
            if finger_position[6] > 0:
                mixing_score += energy_cost_index
            if finger_position[5] > 0:
                mixing_score += energy_cost_thumb

            # Estimate hand position
            # LEFT HAND
            left_hand_position = []
            if finger_position[0] > 0:
                left_hand_position.append(finger_position[0]+2)
            if finger_position[1] > 0:
                left_hand_position.append(finger_position[1]+1)
            if finger_position[2] > 0:
                left_hand_position.append(finger_position[2])
            if finger_position[3] > 0:
                left_hand_position.append(finger_position[3]-1)
            if finger_position[4] > 0:
                left_hand_position.append(finger_position[4]-2)
            # RIGHT HAND
            right_hand_position = []
            if finger_position[5] > 0:
                right_hand_position.append(finger_position[5]+2)
            if finger_position[6] > 0:
                right_hand_position.append(finger_position[6]+1)
            if finger_position[7] > 0:
                right_hand_position.append(finger_position[7])
            if finger_position[8] > 0:
                right_hand_position.append(finger_position[8]-1)
            if finger_position[9] > 0:
                right_hand_position.append(finger_position[9]-2)

            if len(left_hand_position) > 0:
                left_hand_position = int(np.array(left_hand_position).mean())
            else:
                left_hand_position = previous_frame_left_hand_position
            if len(right_hand_position) > 0:
                right_hand_position = int(np.array(right_hand_position).mean())
            else:
                right_hand_position = previous_frame_right_hand_position

            dist = previous_frame_left_hand_position - left_hand_position
            if dist < 0:
                dist = dist*-1
            left_hand_distance += dist
            dist = previous_frame_right_hand_position - right_hand_position
            if dist < 0:
                dist = dist*-1
            right_hand_distance += dist

            if left_hand_position > right_hand_position:
                mixing_score +=5

            previous_frame_left_hand_position = left_hand_position
            previous_frame_right_hand_position = right_hand_position

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
    new_specimens = np.empty((specimens_count, frames.shape[0], 88), dtype=int)
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
