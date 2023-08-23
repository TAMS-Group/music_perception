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
plt.switch_backend('agg')

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

        self.onset_delta = rospy.get_param("~onset_delta", 3.0)
        self.ctx_pre = rospy.get_param("~ctx_pre", 0.3)
        self.ctx_pre_hops = int(self.ctx_pre*self.sr/self.hop_length)
        self.ctx_post = rospy.get_param("~ctx_post", 0.3)
        self.ctx_post_hops = int(self.ctx_post*self.sr/self.hop_length)

        self.perceptual_weighting = rospy.get_param("~perceptual_weighting", True)
        self.log_max_raw_cqt = rospy.get_param("~log_max_raw_cqt", False)

        # mechanism to compensate for clock drift in the /audio_stamped topic
        # ideally this should be compensated in the audio driver, but this is a local workaround
        # as I'm unsure how well the concept generalizes
        self.drift_per_hour = rospy.Duration(rospy.get_param("~drift_s_per_hour", 0.0))
        self.startup_time = rospy.Time()

        # number of samples for analysis window
        self.window_t = rospy.get_param("~window_size", 1.0)
        # and overlap regions between consecutive windows
        self.window_overlap_t = rospy.get_param("~window_overlap", 0.5)
        # length of image spectrum
        self.spectrum_length_t = rospy.get_param("~spectrum_length", 2.0)

        self.window = int(self.sr * self.window_t)
        self.window_overlap = int(self.sr * self.window_overlap_t)
        self.spectrum_length = int(self.sr * self.spectrum_length_t / self.hop_length)

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

        rospy.loginfo(
            f"Onset detector ready using parameters:\n"
            f"min_note: {self.min_note}\n"
            f"max_note: {self.max_note}\n"
            f"semitones_above: {semitones_above}\n"
            f"reference_amplitude: {self.reference_amplitude}\n"
            f"loudest_expected_db: {self.loudest_expected_db}\n"
            f"onset_delta: {self.onset_delta}\n"
            f"drift_s_per_hour: {self.drift_per_hour.to_sec()}\n"
            f"perceptual_weighting: {self.perceptual_weighting}\n"
            f"window_size: {self.window_t}\n"
            f"window_overlap: {self.window_overlap_t}\n"
            f"spectrum_length: {self.spectrum_length_t}\n"
        )

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
        self.pub_envelope = rospy.Publisher(
            "~envelope", Image, queue_size=1, tcp_nodelay=True
        )

        self.pub_drift = rospy.Publisher(
            "~drift", Float32, queue_size=1
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
        self.current_onsets = []

        self.last_envelope = None

    def harmonics_for_cqt_index(self, fundamental_note_idx):
        return np.array([fundamental_note_idx + i for i in (0, 12, 19, 24, 28, 31, 35, 36) if fundamental_note_idx + i < self.semitones])

    @property
    def current_drift(self) -> rospy.Duration:
        return self.drift_per_hour * (self.buffer_time - self.startup_time).to_sec() / 3600.0

    def publish_spectrogram(self, spec, onsets):
        if self.pub_spectrogram.get_num_connections() == 0:
            self.spectrogram = None
            return

        # throw away overlap
        spec = spec[:, self.overlap_hops:-self.overlap_hops]
        spec_length_hops = spec.shape[1]
        spec_length_t = spec_length_hops * self.hop_length / self.sr

        spectrogram_offset = self.spectrogram.shape[1] if self.spectrogram is not None else 0

        self.current_onsets += [int((o - self.window_overlap_t) * self.sr / self.hop_length) + spectrogram_offset for o in onsets]

        if self.spectrogram is None:
            self.spectrogram = spec
        else:
            self.spectrogram = np.concatenate([self.spectrogram, spec], 1)
            if self.spectrogram.shape[1] > self.spectrum_length:
                drop_length = self.spectrogram.shape[1] - self.spectrum_length
                self.spectrogram = self.spectrogram[:, drop_length:]
                self.current_onsets = [o - drop_length for o in self.current_onsets if o >= spec_length_hops]

        # cut noise floor and normalize spectrogram in uint8
        spectrogram = np.maximum(0.0, self.spectrogram)
        upper_bound = max(self.loudest_expected_db, np.max(spectrogram))
        spectrogram = np.array(
            spectrogram*255 / upper_bound, dtype=np.uint8
        )

        heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
        LINECOLOR = [255, 0, 255]
        for o in self.current_onsets:
            heatmap[:, o][:] = LINECOLOR

        self.pub_spectrogram.publish(
            self.cv_bridge.cv2_to_imgmsg(heatmap, "bgr8")
            )

    def publish_envelope(self, envelope):
        if self.last_envelope is not None and self.pub_envelope.get_num_connections() > 0:
            fig = plt.figure(dpi= 300)
            fig.gca().set_title("Onset envelope")
            fig.gca().plot(np.concatenate((
                    self.last_envelope,
                    envelope[self.overlap_hops:-self.overlap_hops]
                )))
            fig.gca().set_ylim(0, np.max((4, np.max(envelope))))
            #fig.gca().axhline(1.0, 0, envelope, color="red")
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            env_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            plt.close(fig)
            self.pub_envelope.publish(self.cv_bridge.cv2_to_imgmsg(env_img, "rgb8"))
        self.last_envelope = envelope[self.overlap_hops:-self.overlap_hops]

    def fundamental_frequency_for_onset(self, onset):
        # sum at most self.window_overlap to make sure the data exists
        prediction_averaging_window = self.ctx_post # prediction window
        transient_duration = 0.06 # seconds / expected maximum length of transient transient

        excerpt = self.buffer[
            int((onset+transient_duration) * self.sr):
                int((onset+transient_duration+prediction_averaging_window) * self.sr)
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
        msg.hop_length = self.hop_length
        msg.sample_rate = self.sr

        msg.header.stamp = \
            self.buffer_time + rospy.Duration(self.window_overlap_t) + self.current_drift
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

        if self.log_max_raw_cqt:
            rospy.loginfo(f"max cqt: {np.max(cqt)}")

        cqt_db = librosa.amplitude_to_db(cqt, ref=self.reference_amplitude)

        if self.perceptual_weighting:
            # TODO: possibly use https://github.com/keunwoochoi/perceptual_weighting , but that requires SPL
            cqt_frequencies = librosa.cqt_frequencies(cqt.shape[0], fmin= self.min_freq)
            cqt_db+= librosa.frequency_weighting(cqt_frequencies, kind= 'A')[:, np.newaxis]

        cqt_db = np.maximum(0.0, cqt_db)

        return cqt_db

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
        elif self.buffer_time is not None and (time_difference := now - (self.buffer_time + rospy.Duration(self.buffer.shape[0]/self.sr))) > rospy.Duration(0.005):
            rospy.logerr_throttle(5.0, f"lost audio samples worth {time_difference.to_sec()}s. will drop remaining buffer and start over.")
            self.reset()
        if seq > self.last_seq + 1 and not self.first_input:
            jump = seq-self.last_seq
            rospy.logwarn(
                f"ros message drop detected: seq jumped "
                f"from {self.last_seq} to {seq} "
                f"(difference of {jump})"
            )
            end_of_buffer_time = \
                self.buffer_time + rospy.Duration(self.buffer.shape[0]/self.sr)
            self.reset()
            self.buffer_time = \
                end_of_buffer_time + rospy.Duration((jump-1) * (len(msg_data)/self.sr))

        self.last_time = now
        self.last_seq = seq

        # take time from message headers and increment based on data
        if self.buffer_time is None:
            self.buffer_time = now
            if self.first_input:
                self.startup_time = now

        self.first_input = False

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
            sr=self.sr, S=cqt, aggregate= np.median
        )
        onsets_cqt_raw = librosa.onset.onset_detect(
            y=self.buffer,
            sr=self.sr,
            hop_length=self.hop_length,
            onset_envelope=onset_env_cqt,
            units="time",
            backtrack=False,
            # wait= 0.1*self.sr/self.hop_length,
            delta=self.onset_delta, # TODO: scale delta as 1.96 * stddev of last seconds
            normalize=False,
            pre_max= self.ctx_pre_hops,
            post_max= self.ctx_post_hops,
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

        self.publish_spectrogram(cqt, onsets_cqt)
        self.publish_envelope(onset_env_cqt)

        # publish events and plot visualization
        for o in onsets_cqt:
            fundamental_frequency, confidence = \
                self.fundamental_frequency_for_onset(o)
            t = self.buffer_time + self.current_drift + rospy.Duration(o)

            no = NoteOnset()
            no.header.stamp = t
            if fundamental_frequency != 0.0:
                no.note = librosa.hz_to_note(fundamental_frequency)
                no.confidence = confidence

                # look at ctx_post after onset to determine maximum loudness
                onset_hop = int(o * self.sr / self.hop_length)
                note_idx= librosa.note_to_midi(no.note) - self.min_midi
                onset_harmonics = cqt[self.harmonics_for_cqt_index(note_idx), onset_hop:onset_hop+self.ctx_post_hops]
                loudness_dba = np.log(np.exp(onset_harmonics).sum(axis=0))
                max_idx = loudness_dba.argmax()
                no.loudness = loudness_dba[max_idx]
                no.spectrum = onset_harmonics[:, max_idx]

                rospy.loginfo(f"at {t.to_sec():.4F} found conf. {no.confidence:.2f} / note {no.note:>2} / vol {no.loudness:.2f}dB")

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
        if compute_time > rospy.Duration(self.window_t):
            rospy.logerr(f"computation took longer than processed window (processed {self.window_t:.2F}s, took {compute_time.to_sec():.2F}s)")
        self.pub_drift.publish(self.current_drift.to_sec())


def main():
    rospy.init_node("detect_onset")

    detector = OnsetDetector()
    detector.start()
    rospy.spin()


if __name__ == "__main__":
    main()
