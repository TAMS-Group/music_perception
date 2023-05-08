from collections import namedtuple
import librosa
from mido import MidiFile
from music_perception.msg import Piece, NoteOnset
import rospy
from std_msgs.msg import Header
import subprocess
import sys
from tempfile import TemporaryDirectory
import os

__all__ = ['piece_from_midi', 'piece_from_ly']

def piece_from_ly(lilypond_string : str) -> Piece:
    '''
    Run lilypond to generate a midi file from a lilypond string
    and convert it to a Piece message via `piece_from_midi`.

    @param lilypond_string: a string containing lilypond notation    
    '''
    dir= TemporaryDirectory(prefix='ly2onsets-')
    with open(os.path.join(dir.name, 'score.ly'), 'w') as ly_file:
        ly_file.write(lilypond_string)
    result = subprocess.run(['lilypond', 'score.ly'], cwd=dir.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stdout.decode('utf-8'))
    return piece_from_midi(os.path.join(dir.name, 'score.midi'))

def piece_from_midi(midi_path : str) -> Piece:
    '''
    Reads a midi file and returns a Piece message with all note onsets and their timing.
    Notice that the loudness values are not scaled to dB, but are MIDI velocity values 0-127,
    which have to be mapped to reasonable values for different instruments.
    https://www.cs.cmu.edu/~rbd/papers/velocity-icmc2006.pdf

    @param midi_path: path to the midi file
    @return: a Piece message with all note onsets and their timing
    '''
    piece = Piece()
    with MidiFile(midi_path) as midi:
        now = 0.0
        ongoing = {}

        NoteOn = namedtuple('NoteOn', ['time', 'note', 'velocity'])
        for on in filter(lambda m: m.type == 'note_on', iter(midi)):
            now += on.time
            if on.velocity == 0:
                if on.note not in ongoing:
                    continue
                piece.onsets.append(
                    NoteOnset(
                        header= Header(stamp= rospy.Time(ongoing[on.note].time)),
                        note = librosa.midi_to_note(on.note),
                        # These are MIDI velocity values 0-127, not loudness (and not dB scaled)
                        loudness= ongoing[on.note].velocity,
                        duration= rospy.Duration(now - ongoing[on.note].time)
                    )
                )
                del ongoing[on.note]
            else: # on.velocity > 0
                ongoing[on.note] = NoteOn(time= now, note= on.note, velocity= on.velocity)
    return piece
