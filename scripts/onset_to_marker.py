#!/usr/bin/env python

import rospy

from visualization_msgs.msg import MarkerArray, Marker
from music_perception.msg import NoteOnset
from std_msgs.msg import ColorRGBA

from librosa import note_to_midi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class OnsetToMarker:
    def __init__(self):
        self.min_note = rospy.get_param("min_note")
        self.min_midi = note_to_midi(self.min_note)
        self.max_note = rospy.get_param("max_note")
        self.max_midi = note_to_midi(self.max_note)

        # one hsv range per octave
        hsv = plt.get_cmap("hsv")
        octaves = int(np.ceil((self.max_midi - self.min_midi)/12))
        self.cmap = ListedColormap(np.tile(hsv(np.linspace(0, 1, int(np.ceil(256/octaves)))), (octaves,1) )[0:256])
        self.cmap.set_bad((0, 0, 0, 1))  # make sure they are visible

    def start(self):
        self.sub_onset = rospy.Subscriber(
            "onsets",
            NoteOnset,
            self.onset_cb,
            queue_size=100,
            tcp_nodelay=True
        )
        self.pub_markers = rospy.Publisher(
            "onsets_markers",
            MarkerArray,
            queue_size=100,
            tcp_nodelay=True
        )

    def onset_cb(self, msg):
        markers = MarkerArray()

        m = Marker()
        if msg.note != '':
            m.ns = msg.note
        else:
            m.ns = "unknown"
        m.action = Marker.ADD

        m.header = msg.header
        if m.header.frame_id == '':
            # stub value if none is set for onset
            m.header.frame_id = "onset_frame"

        m.type = Marker.SPHERE

        m.pose.orientation.w = 1.0

        m.scale.x = 0.005
        m.scale.y = m.scale.x
        m.scale.z = m.scale.x

        if msg.note != '':
            m.color = ColorRGBA(
                *self.cmap(
                    (note_to_midi(msg.note) - self.min_midi) /
                    (self.max_midi - self.min_midi)
                )
            )
        else:
            m.color = ColorRGBA(*self.cmap.get_bad())

        markers.markers.append(m)
        self.pub_markers.publish(markers)


def main():
    rospy.init_node("onset_to_marker")

    otm = OnsetToMarker()
    otm.start()
    rospy.spin()


if __name__ == "__main__":
    main()
