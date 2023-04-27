#!/usr/bin/env python

# This script reads a lilypond notation file and publishes the onsets of all notes
# author: Michael 'v4hn' Goerner, 2023

from collections import namedtuple
import librosa
from mido import MidiFile, MetaMessage, MidiTrack
from music_perception.msg import NoteOnset, Piece
import os.path
import rospy
import std_msgs.msg
import subprocess
import sys
from tempfile import TemporaryDirectory

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input.ly>")
    sys.exit(1)

ly= open(sys.argv[1], 'r').read()

dir= TemporaryDirectory(prefix='ly2onsets-')
with open(os.path.join(dir.name, 'score.ly'), 'w') as ly_file:
    ly_file.write(ly)

subprocess.run(['lilypond', 'score.ly'], cwd=dir.name)
with MidiFile(os.path.join(dir.name, 'score.midi')) as midi:
    piece = Piece()
    now = 0.0
    ongoing = {}

    NoteOn = namedtuple('NoteOn', ['time', 'note', 'velocity'])
    for on in filter(lambda m: m.type == 'note_on', iter(midi)):
        now += on.time
        if on.velocity == 0:
            if on.note not in ongoing:
                rospy.logwarn(f"Found note_off without note_on, ignoring")
            else:
                piece.onsets.append(
                    NoteOnset(
                        header= std_msgs.msg.Header(stamp= rospy.Time(ongoing[on.note].time)),
                        note = librosa.midi_to_note(on.note),
                        # TODO: The linear velocity values 0-255 should be transferred to a dB scale
                        #       At the very least the subscriber needs to scale the values
                        loudness= ongoing[on.note].velocity,
                        duration= rospy.Duration(now - ongoing[on.note].time)
                    )
                )
                del ongoing[on.note]
        else: # on.velocity > 0
            ongoing[on.note] = NoteOn(time= now, note= on.note, velocity= on.velocity)
    
    rospy.init_node('ly2piece')
    pub = rospy.Publisher('piece', Piece, queue_size= 1, latch= True)
    pub.publish(piece)
    rospy.spin()
