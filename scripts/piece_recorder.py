#!/usr/bin/env python

import os
import pickle
import rospkg
import rospy
from music_perception.msg import NoteOnset
from music_perception.msg import Piece
from std_srvs.srv import SetBool

class PieceRecorder:
    def __init__(self):
        self.piece= Piece()
        self.recording= False
        self.storage_path = rospy.get_param("~storage_path", rospkg.RosPack().get_path("music_perception") + "/pieces")

        self.piece_sub= rospy.Subscriber("onsets", NoteOnset, self.onset_cb)
        self.service= rospy.Service("recording", SetBool, self.recording_cb)
        self.piece_pub= rospy.Publisher("recorded_piece", Piece, queue_size=1)

    def recording_cb(self, req):
        if not req.data and len(self.piece.onsets) > 0:
            self.recording = False
            piece_path = os.path.join(self.storage_path, 'piece_' + str(rospy.Time.now().to_sec()) + '.pkl')
            pickle.dump(self.piece, file = open(piece_path, 'wb'))
            self.piece_pub.publish(self.piece)
            self.piece = Piece()
            return {'success': True, 'message': f"Saved piece as '{piece_path}'"}
        elif self.recording == False and req.data:
            self.recording = True
            return {'success': True, 'message': 'Recording piece'}

        return {'success': False, 'message': 'Nothing to do'}

    def onset_cb(self, onset : NoteOnset):
        if self.recording == False:
            return

        if onset.note != '':
            if len(self.piece.onsets) == 0:
                self.piece.header = onset.header
            self.piece.onsets.append(onset)

def main():
    rospy.init_node("piece_recorder")
    p = PieceRecorder()
    rospy.spin()

if __name__ == "__main__":
    main()
