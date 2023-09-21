#!/usr/bin/env python

# This script reads a lilypond notation file and publishes the onsets of all notes
# author: Michael 'v4hn' Goerner, 2023

from music_perception.msg import Piece
from music_perception.topiece import piece_from_ly
import os.path
import rospy
import subprocess
import sys
from tempfile import TemporaryDirectory

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} {{<input.ly>|lilypond-string}}")
    sys.exit(1)

try:
    ly= open(sys.argv[1], 'r').read()
except FileNotFoundError:
    # assume argv[1] is a core lilypond string, e.g., "d4\pppp e fis2 a4\mf fis2 e4 d4"
    ly= R'\version "2.20.0"' R"\score { \unfoldRepeats { \relative c' { " + str(sys.argv[1]) + " } } \midi { tempo = 60 } }"

piece = piece_from_ly(ly)
rospy.init_node('ly2piece')
pub = rospy.Publisher('piece_midi_loudness', Piece, queue_size= 1, latch= True)
pub.publish(piece)
rospy.spin()
