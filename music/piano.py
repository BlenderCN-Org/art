import numpy as np
import random as rd


class finger:
    def __init__(self, index):
        self.index = index
        self.is_pressing = False
        self.position = 0

    def get_position(self):
        return self.position

    def get_pressing(self):
        return self.is_pressing

    def set_position(self, key_index):
        self.position = key_index



class hand:
    def __init__(self, left):
        self.left = left
        self.position = 0
        self.resting = True
        self.fingers = []
        for finger_index in range(0, 5):
            self.fingers.append(finger(finger_index))

    def get_position(self):
        return self.position

    def get_resting(self):
        return self.resting

    def set_position(self, key_index):
        self.position = key_index

    def set_finger_positions(self, positions):
        for index in range(0, 5):
            self.fingers[index].set_position(positions[index])

    def get_finger_on_key(self, key_index):
        for finger_index in range(0, 5):
            if self.fingers[finger_index].get_position() == key_index:
                return finger_index
        return None



class frame:
    def __init__(self, frame_info, previous_frame):
        self.left_hand  = hand(True)
        self.right_hand = hand(False)
        self.keys = frame_info
        self.previous_frame = previous_frame
        self.pressed_keys = []
        for key_index in range(0, self.keys.shape[0]):
            if self.keys[key_index] > 0:
                self.pressed_keys.append(key_index)

        self.reset_hands_position()

    def get_finger_on_key(self, index):
        if index < self.keys.shape[0]:
            on_left_hand = self.left_hand.get_finger_on_key(index)
            on_right_hand = self.right_hand.get_finger_on_key(index)
            if on_left_hand is not None:
                return on_left_hand
            elif on_right_hand is not None:
                return on_right_hand + 5
            else:
                return None
        else:
            return None

    def get_pressed_keys(self, index):
        return self.pressed_keys

    def is_pressed(self, index):
        if index in self.pressed_keys:
            return True
        else:
            return False

    def reset_hands_position(self):
        # Will the hands be inverted?
        # Select hand boundary
        print ()
        if self.previous_frame is not None:
            print (self.previous_frame.pressed_keys)
        print (self.pressed_keys)

        if self.previous_frame is not None:
            for key_index in self.pressed_keys:
                if key_index in self.previous_frame:
                    finger = self.get_finger_on_key(key_index)
                    print (finger)

        if len(self.pressed_keys) == 0:
            self.left_hand.set_finger_positions([0, 0, 0, 0, 0])
            self.right_hand.set_finger_positions([0, 0, 0, 0, 0])
        elif len(self.pressed_keys) == 1:
            random_finger = rd.randint(0, 4)
            positions = [0, 0, 0, 0, 0]
            positions[random_finger] = self.pressed_keys[0]
            if rd.random() > 0.5:
                self.left_hand.set_finger_positions(positions)
            else:
                self.right_hand.set_finger_positions(positions)
        else:
            random_boundary = rd.randint(0, len(self.pressed_keys))
            positions_a = [0, 0, 0, 0, 0]
            positions_b = [0, 0, 0, 0, 0]
            for pressed_key_index in range(0, len(self.pressed_keys)):
                # TODO should be ordered from left to right
                random_finger = rd.randint(0, 4)
                if pressed_key_index < random_boundary:
                    positions_a[random_finger] = self.pressed_keys[pressed_key_index]
                else:
                    positions_b[random_finger] = self.pressed_keys[pressed_key_index]
            if rd.random() > 0.5:
                self.left_hand.set_finger_positions(positions_a)
                self.right_hand.set_finger_positions(positions_b)
            else:
                self.right_hand.set_finger_positions(positions_a)
                self.left_hand.set_finger_positions(positions_b)



class batch:
    def __init__(self, reduced_frames, start, end, previous_batches):
        self.batch_reduced_frames = np.array(reduced_frames[start:end])
        self.start = start
        self.end = end
        self.previous_batches = previous_batches
        self.frame_objects = []
        self.batch_size = end - start
        for frame_index in range(0, self.batch_size):
            previous_frame = None
            if frame_index == 0:
                if len(self.previous_batches) > 0:
                    previous_frame = self.previous_batches[-1].frame_objects[-1]
            else:
                previous_frame = self.frame_objects[-1]
            self.frame_objects.append(frame(self.batch_reduced_frames[frame_index], previous_frame))

    def get_frame(self, index):
        if index < self.batch_reduced_frames.shape[0]:
            return self.batch_reduced_frames[index]
        else:
            return None

    def get_fitness(self, previous_batches, condensed_frame_index):
        return 3



class specimen:
    def __init__(self, full_frames, reduced_frames, condensed_frame_index, batch_size):
        self.full_frames = full_frames
        self.reduced_frames = reduced_frames
        self.condensed_frame_index = condensed_frame_index
        self.batches = []
        self.batch_size = batch_size
        start = 0
        previous_batches = []
        while 1:
            end = start + batch_size
            if end > self.reduced_frames.shape[0]-1:
                end = self.reduced_frames.shape[0]-1
            self.batches.append(batch(self.reduced_frames, start, end, previous_batches))
            previous_batches.append(self.batches[-1])
            start += batch_size
            if end == self.reduced_frames.shape[0]-1:
                break

    def get_batch(self, index):
        if index < len(self.batches) - 1:
            return self.batches[index]
        else:
            return None

    def get_fitness(self):
        fitness_list = []
        for batch in self.batches:
            fitness_list.append (batch.get_fitness(self.condensed_frame_index))
        return fitness_list
