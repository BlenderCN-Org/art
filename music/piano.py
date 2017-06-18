class finger:
    def __init__(index):
        self.index = index
        self.is_pressing = False
        self.position = 0

    def get_position():
        return self.position

    def get_pressing():
        return self.is_pressing

    def set_position(key_index):
        self.position = key_index



class hand:
    def __init__(left):
        self.left = left
        self.position = 0
        self.resting = True

    def get_position():
        return self.position

    def get_resting():
        return self.resting

    def set_position(key_index):
        self.position = key_index



class frame:
    def __init__():
        self.left_hand  = hand(True)
        self.right_hand = hand(False)
        self.keys = []

    def get_key(index):
        return self.keys[index]

    def get_pressed_keys(index):
        return self.keys[index]



class batch:
    def __init__(reduced_frames):
        self.reduced_frames = reduced_frames

    def get_frame(index):
        return self.reduced_frames[index]



class specimen:
    def __init__(reduced_frames, batch_size):
        self.batches = []

    def get_batch(index):
        return self.batches[index]
