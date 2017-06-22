import re
import os
import json

import numpy as np



def open_midi(midigram_filepath, octave_shift):
    ## Midigram collumns
    # 1: On (MIDI pulse units)
    # 2: Off (MIDI pulse units)
    # 3: Track number
    # 4: Channel number
    # 5: Midi pitch -> 60 -> c4 / 61 -> c#4
    # 6: Midi velocity

    #midi_pulse_per_second  = 480.299
    #fps = 30
    #fps = 11.7 #TODO translate correctly Pulse to FPS
    # 0,934579439
    beats_per_minute = 120.0
    pulse_per_quarter = 559.0
    midi_pulse_lenght_ms = (60000.0 / (beats_per_minute * pulse_per_quarter)) # 2.604
    fps = 60.0
    frame_lenght_ms = 1000.0 / fps # 16,666666667
    margin = 1
    keyboard_size = 88
    max_frames = 10000

    # 72719
    # 3950

    # 3891
    # 3885

    #print (frame_lenght_ms / midi_pulse_lenght_ms)
    #print (72480.0 / (frame_lenght_ms / midi_pulse_lenght_ms))

    midi_pulse_per_frame = frame_lenght_ms / midi_pulse_lenght_ms

    with open (midigram_filepath, "r") as midigram_file:
        data=midigram_file.readlines()

    last_frame = 0
    midi_frames = {}
    for data_line in data[:-1]:
        line_regex = re.search(r'^(?P<ontime>\d+)\s(?P<offtime>\d+)\s(?P<track>\d+)\s(?P<channel>\d+)\s(?P<pitch>\d+)\s(?P<velocity>\d+)', data_line)
        #ontime = int(float(line_regex.group("ontime"))/midi_pulse_per_frame)
        #offtime = int(float(line_regex.group("offtime"))/midi_pulse_per_frame)
        ontime = int(line_regex.group("ontime"))
        offtime = int(line_regex.group("offtime"))
        track = int(line_regex.group("track"))
        channel = int(line_regex.group("channel"))
        pitch = int(line_regex.group("pitch"))
        velocity = int(line_regex.group("velocity"))

        if not ontime in midi_frames:
            midi_frames[ontime] = []
        midi_frames[ontime].append({
            "ontime": ontime,
            "offtime": offtime,
            "track": track,
            "channel": channel,
            "pitch": pitch,
            "velocity": velocity,
        })
        if offtime > last_frame:
            last_frame = offtime

    music = np.zeros([last_frame, keyboard_size], dtype=int)

    for mf in midi_frames:
        for mf_note in midi_frames[mf]:
            shifted_pitch = mf_note['pitch']+(octave_shift*12)
            if shifted_pitch >= keyboard_size:
                print ("Skipping note, out of keyboard range")
                continue
            #if mf_note['track'] != 1:
            #    print ("Skipping note, out of track")
            #    continue
            # Fill ontime to offtime values
            for f in range(mf_note['ontime'], mf_note['offtime']):
                music[f][shifted_pitch] = 1

    full_frames = music

    # ENCODE
    # Leave only frames with Key transitions
    # (Delete frame when equal to frame-1).
    frames_filepath_json = "reduced_frames.json"
    if os.path.exists(frames_filepath_json):
        print ("Loading Reduced Frames from file")
        with open(frames_filepath_json) as frames_datafile_json:
            frames = np.array(json.load(frames_datafile_json))
        print ("Reduced frames loaded:", frames.shape[0])
    else:
        frames = np.array(full_frames)
        n = 1
        while n < frames.shape[0]:
            if np.array_equal(frames[n], frames[n-1]):
                frames = np.delete(frames, (n), axis=0)
            else:
                n += 1
        print ("Reduced frames:", frames.shape[0])
        # Save Reduced frames
        print ("Saving Reduced Frames to file")
        with open(frames_filepath_json, 'w', encoding='utf-8') as anim_file:
            json.dump(frames.tolist(), anim_file, sort_keys=True, indent=4, separators=(',', ': '))


    # Map Reduced Frames to Full Frames
    current_fingers_positions = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    condensed_frame_index = 0
    condensed_to_full_mapping = {}
    condensed_pressed_keys = {}
    condensed_used_fingers = {}
    for full_frame_index in range(0, len(full_frames)):
        pressed_keys = []
        for key_index in range(0, 88):
            if full_frames[full_frame_index][key_index] != 0:
                pressed_keys.append(key_index)
        condensed_found = False
        previous_condensed_used_fingers = []
        while not condensed_found:
            try:
                frames[condensed_frame_index]
            except:
                #print ("Missing Condensed Frames")
                break
            condensed_pressed_keys[condensed_frame_index] = []
            condensed_used_fingers[condensed_frame_index] = []
            for key_index in range(0, 88):
                if frames[condensed_frame_index][key_index] != 0:
                    condensed_pressed_keys[condensed_frame_index].append(key_index)
                    condensed_used_fingers[condensed_frame_index].append(frames[condensed_frame_index][key_index])
            if pressed_keys==condensed_pressed_keys[condensed_frame_index]:
                condensed_found = True
                condensed_to_full_mapping[condensed_frame_index] = full_frame_index
                previous_condensed_used_fingers = condensed_pressed_keys[condensed_frame_index]
            else:
                condensed_frame_index += 1

    #notes = [0 for _ in range(0, keyboard_size)]
    #music = [notes for _ in range(0, margin)] + music + [notes for _ in range(0, margin)]
    return music, frames, condensed_frame_index
