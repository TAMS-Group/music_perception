#!/usr/bin/env python

import rospy
import cv_bridge

from audio_common_msgs.msg import AudioDataStamped, AudioInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, ColorRGBA
from music_perception.msg import NoteOnset, CQTStamped

import librosa
import crepe
import crepe.core

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from functools import reduce

import struct
# import time


class OnsetDetector:
    @staticmethod
    def unpack_data(data):
        return np.frombuffer(data, dtype=np.int16).astype(float)

    def check_audio_format(self):
        rospy.loginfo("Waiting for Audio Info")
        info = rospy.wait_for_message("audio_info", AudioInfo)
        if info.channels != 1:
            rospy.logfatal(
                "audio data has more than one channel,"
                "expecting single-channel recording"
            )
        elif info.sample_rate != 44100:
            rospy.logfatal(
                f"sample rate {info.sample_rate} is not 44100"
                )
        elif info.sample_format != "S16LE":
            rospy.logfatal(
                f"sample format '{info.sample_format}' is not S16LE"
                )
        elif info.coding_format != "wave":
            rospy.logfatal(
                f"coding '{info.coding_format}' is not raw"
                )
        else:
            rospy.loginfo("Audio compatible")
            return True
        return False

    def __init__(self):
        self.sr = 44100
        self.hop_length = 512

        self.min_note = rospy.get_param("min_note")
        self.min_freq = librosa.note_to_hz(self.min_note)
        self.min_midi = librosa.note_to_midi(self.min_note)

        self.max_note = rospy.get_param("max_note")
        self.max_freq = librosa.note_to_hz(self.max_note)
        self.max_midi = librosa.note_to_midi(self.max_note)

        # how many semitones above the highest note to include to analyze harmonics (recommend 24+eps for 3 overtones+wiggle room)
        semitones_above = rospy.get_param("~semitones_above")
        self.semitones = semitones_above + self.max_midi - self.min_midi

        # if provided, db values will be given relative to this amplitude value
        self.reference_amplitude = rospy.get_param("~reference_amplitude", np.inf)
        self.loudest_expected_db = rospy.get_param("~loudest_expected_db", 120.0)

        # number of samples for analysis window
        self.window_t = 1.0
        # and overlap regions between consecutive windows
        self.window_overlap_t = 0.5

        self.window = int(self.sr * self.window_t)
        self.window_overlap = int(self.sr * self.window_overlap_t)

        self.overlap_hops = int(self.window_overlap / self.hop_length)

        # preload model to not block the callback on first message
        # capacities: 'tiny', 'small', 'medium', 'large', 'full'
        self.crepe_model = "full"
        crepe.core.build_and_load_model(self.crepe_model)

        if not self.check_audio_format():
            rospy.signal_shutdown("incompatible audio format")
            return

        self.last_time = rospy.Time.now()
        self.last_seq = 0

        self.cv_bridge = cv_bridge.CvBridge()

        self.buffer = np.array([0.0]*self.sr, dtype=float)
        # warm up classifier / jit caches
        _ = self.cqt()
        _ = self.fundamental_frequency_for_onset(0.0)
        self.reset()

        self.first_input = True

    def start(self):
        self.pub_spectrogram = rospy.Publisher(
            "spectrogram", Image, queue_size=1, tcp_nodelay=True
        )

        self.pub_compute_time = rospy.Publisher(
            "~compute_time", Float32, queue_size=1, tcp_nodelay=True
        )

        self.pub_cqt = rospy.Publisher(
            "cqt", CQTStamped, queue_size=100, tcp_nodelay=True
        )
        self.pub_onset = rospy.Publisher(
            "onsets", NoteOnset, queue_size=100, tcp_nodelay=True
        )

        self.sub = rospy.Subscriber(
            "audio_stamped",
            AudioDataStamped,
            self.audio_cb,
            queue_size=500,
            tcp_nodelay=True,
        )

    def reset(self):
        # audio buffer
        self.buffer_time = None
        self.buffer = np.array([], dtype=float)

        # visualization
        self.spectrogram = None
        self.previous_onsets = []

    def update_spectrogram(self, spec, onsets):
        if self.pub_spectrogram.get_num_connections() == 0:
            self.spectrogram = None
            return

        # throw away overlap
        spec = spec[:, self.overlap_hops:-self.overlap_hops]
        onsets = [o - self.window_overlap_t for o in onsets]

        if self.spectrogram is None:
            self.spectrogram = spec
            return
        elif self.spectrogram.shape[1] > spec.shape[1]:
            self.spectrogram = self.spectrogram[:, -spec.shape[1]:]
        self.spectrogram = np.concatenate([self.spectrogram, spec], 1)

        # cut noise floor and normalize spectrogram in uint8
        spectrogram = np.maximum(0.0, self.spectrogram)
        upper_bound = max(self.loudest_expected_db, np.max(spectrogram))
        spectrogram = np.array(
            spectrogram*255 / upper_bound, dtype=np.uint8
        )

        heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
        LINECOLOR = [255, 0, 255]
        for o in self.previous_onsets:
            heatmap[:, int(o * self.sr / self.hop_length)][:] = LINECOLOR
        for o in onsets:
            heatmap[
                :,
                int(self.window / self.hop_length +
                    o * self.sr / self.hop_length)
            ][:] = LINECOLOR
        self.previous_onsets = onsets

        self.pub_spectrogram.publish(
            self.cv_bridge.cv2_to_imgmsg(heatmap, "bgr8")
            )

    def fundamental_frequency_for_onset(self, onset):
        prediction_averaging_window = (
            0.1 * self.sr
        )  # at most self.window_overlap to make sure the data exists
        excerpt = self.buffer[
            int(onset * self.sr):
                int(onset * self.sr + prediction_averaging_window)
        ]
        time, freq, confidence, _ = crepe.predict(
            excerpt,
            self.sr,
            viterbi=True,
            model_capacity=self.crepe_model,
            verbose=0
        )

        confidence_threshold = 0.2
        confidence_mask = confidence > confidence_threshold

        thresholded_freq = freq[confidence_mask]
        thresholded_confidence = confidence[confidence_mask]
        if len(thresholded_freq) > 0:
            buckets = {}
            for f, c in zip(thresholded_freq, thresholded_confidence):
                note = librosa.hz_to_note(f)
                buckets[note] = buckets.get(note, []) + [c]

            def add_confidence(note):
                return reduce(lambda x, y: x + y, buckets.get(note))
            winner = max(buckets, key=lambda a: add_confidence(a))
            winner_freq = librosa.note_to_hz(winner)
            return winner_freq, max(buckets[winner])
        else:
            return 0.0, 0.0

    def publish_cqt(self, cqt):
        msg = CQTStamped()
        msg.number_of_semitones = self.semitones
        msg.min_note = self.min_note
        msg.hop_length = rospy.Duration(self.hop_length / self.sr)

        msg.header.stamp = \
            self.buffer_time + rospy.Duration(self.window_overlap_t)
        msg.data = \
            cqt[:, self.overlap_hops:-self.overlap_hops].flatten(order="F")
        self.pub_cqt.publish(msg)

    def cqt(self):
        cqt = np.abs(
            librosa.cqt(
                y=self.buffer,
                sr=self.sr,
                hop_length=self.hop_length,
                fmin=self.min_freq,
                n_bins=self.semitones,
            )
        )
        # rospy.loginfo(f"max cqt: {np.max(cqt)}")
        return librosa.amplitude_to_db(cqt, ref=self.reference_amplitude)

    def audio_cb(self, msg):
        now = msg.header.stamp
        seq = msg.header.seq

        msg_data = OnsetDetector.unpack_data(msg.audio.data)

        # handle bag loop graciously
        if now < self.last_time:
            rospy.loginfo("detected bag loop")
            self.reset()
        elif seq != self.last_seq + 1 and not self.first_input:
            rospy.logwarn(f"something weird happened, seq jumped from {self.last_seq} to {seq}")
        elif self.buffer_time is not None and (time_difference := self.buffer_time + rospy.Duration(self.buffer.shape[0]/self.sr) - now) > rospy.Duration(0.005):
            rospy.logwarn(f"lost audio samples worth {time_difference.to_sec()}s. will drop remaining buffer and start over.")
            self.reset()
        if seq > self.last_seq + 1 and not self.first_input:
            jump = seq-self.last_seq
            rospy.logwarn(
                f"sample drop detected: seq jumped "
                f"from {self.last_seq} to {seq} "
                f"(difference of {jump})"
            )
            end_of_buffer_time = \
                self.buffer_time + rospy.Duration(self.buffer.shape[0]/self.sr)
            self.reset()
            self.buffer_time = \
                end_of_buffer_time + rospy.Duration((jump-1) * (len(msg_data)/self.sr))

        self.first_input = False
        self.last_time = now
        self.last_seq = seq

        # take time from message headers and increment based on data
        if self.buffer_time is None:
            self.buffer_time = now

        self.buffer = np.concatenate([
            self.buffer,
            msg_data
            ])

        # aggregate buffer until window+overlaps are full
        if self.buffer.shape[0] < self.window + 2 * self.window_overlap:
            return

        # TODO: it would be nicer to run the computation below asynchronously

        cqt = self.cqt()

        self.publish_cqt(cqt)

        onset_env_cqt = librosa.onset.onset_strength(
            sr=self.sr, S=cqt
        )
        onsets_cqt_raw = librosa.onset.onset_detect(
            y=self.buffer,
            sr=self.sr,
            hop_length=self.hop_length,
            onset_envelope=onset_env_cqt,
            units="time",
            backtrack=False,
            # wait= 0.1*self.sr/self.hop_length,
            delta=4.0,
            normalize=False,
        )

        def in_window(o):
            return (
                o >= self.window_overlap_t and
                o < self.window_t + self.window_overlap_t
                )

        onsets_cqt = [
            o
            for o in onsets_cqt_raw
            if in_window(o)
        ]

        self.update_spectrogram(cqt, onsets_cqt)

        # publish events and plot visualization
        for o in onsets_cqt:
            fundamental_frequency, confidence = \
                self.fundamental_frequency_for_onset(o)
            t = self.buffer_time + rospy.Duration(o)

            no = NoteOnset()
            no.header.stamp = t
            if fundamental_frequency != 0.0:
                no.note = librosa.hz_to_note(fundamental_frequency)
                no.confidence = confidence

                # mean over ~300ms window (~25 samples) after onset
                onset_hop = int(o * self.sr / self.hop_length)
                note_idx= librosa.note_to_midi(no.note) - self.min_midi
                try:
                    no.loudness = cqt[note_idx, onset_hop:onset_hop+25].mean()
                except IndexError:
                    no.loudness = 0.0

                rospy.loginfo(f"found conf. {no.confidence:.2f} / note {no.note:>2} / vol {no.loudness:.2f}dB at {t.to_sec()}")

            self.pub_onset.publish(no)

        if len(onsets_cqt) == 0:
            rospy.logdebug("found no onsets")
        else:
            rospy.logdebug("found {} onsets".format(len(onsets_cqt)))

        # advance buffer, keep one overlap for next processing
        self.buffer_time += rospy.Duration(self.window_t)
        self.buffer = self.buffer[(-2 * self.window_overlap):]

        rospy.loginfo_once("onset detection is online")
        compute_time = rospy.Time.now() - now
        self.pub_compute_time.publish(compute_time.to_sec())
        if compute_time > rospy.Duration(self.window):
            rospy.logerr("computation took longer than processed window")


def main():
    rospy.init_node("detect_onset")

    detector = OnsetDetector()
    detector.start()
    rospy.spin()


if __name__ == "__main__":
    main()
