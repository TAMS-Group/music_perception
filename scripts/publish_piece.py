#!/usr/bin/env python

import pickle
import rospy
import sys
from music_perception.msg import Piece

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: publish_piece.py <path_to_piece>")
        exit(1)
    
    piece = pickle.load(file = open(sys.argv[1], 'rb'))

    rospy.init_node('piece_publisher')
    piece_pub = rospy.Publisher('piece', Piece, queue_size=1, tcp_nodelay=True, latch=True)
    piece_pub.publish(piece)
    rospy.sleep(3.0)