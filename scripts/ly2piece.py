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
    print(f"Usage: {sys.argv[0]} <input.ly>")
    sys.exit(1)

ly= open(sys.argv[1], 'r').read()

piece = piece_from_ly(ly)
rospy.init_node('ly2piece')
pub = rospy.Publisher('piece_midi_loudness', Piece, queue_size= 1, latch= True)
pub.publish(piece)
rospy.spin()
