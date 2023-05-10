\version "2.24.1"
\include "swing.ly"

% original phrase in C Major
first_phrase = \relative { c'4 d e f g2 g a4 a a a g1 }

second_voice = \relative { c'4 b c d e2 e2 f4 f f f e1 }

first_phrase_two_voices = << \first_phrase \\ \second_voice >>

first_phrase_chords = \relative { c'4 <d b> <e c> <f d> <g e>2 <g e> <a f>4 <a f> <a f> <a f> <g e>1 }

% transposed by a Major third to E Major
first_phrase_e_major = \transpose c' e' \first_phrase

% modulations between diatonic modes can be done relatively easy by choosing the fitting base note for the new mode,
% write down the melody with "renamed" notes and transposing the result to the base tone (here always c') afterwards
% https://en.wikipedia.org/wiki/Diatonic_scale
first_phrase_minor =   \transpose a' c' \relative { a'4 b c d e2 e f4 f f f e1 }
first_phrase_dorian =  \transpose d' c'  \relative { d'4 e f g a2 a b4 b b b a1 }
first_phrase_locrian = \transpose b' c' \relative { b'4 c d e f2 f g4 g g g f1 }

first_phrase_swinged = {
  % wrap it in the magic line to turn it into swinged rhythm
  \tempo \markup { Swing \rhythm { 8[ 8] } = \rhythm { \tuplet 3/2 { 4 8 } } } \applySwing 8 #'(3 2) {
    \relative { c'8 c d d e e f f g4 g g g a8 a a a a a a a g2 g2 }
  }
  \tempo \markup { \rhythm { 8[ 8] } = \rhythm { 8 8 } }
}

end_of_phrase = { \bar "||" \break }


\header {
  title = "Modifying ''All my ducklings''"
}

\score {
  \new Staff {
    \tempo 4 = 160
    \key c \major

    \textMark "C Major"
    \first_phrase
    \end_of_phrase

    \textMark "Two voices"
    \first_phrase_two_voices
    \end_of_phrase
    
    \textMark "Chord notation (otherwise same as above)"
    \first_phrase_chords
    \end_of_phrase

    \first_phrase_swinged
    \end_of_phrase

    \textMark "E Major"
    \key e \major
    \first_phrase_e_major
    \end_of_phrase

    \textMark "C Minor"
    \key c \minor
    \first_phrase_minor
    \end_of_phrase

    \textMark "C Dorian"
    \key c \major
    \first_phrase_dorian
    \end_of_phrase

    \textMark "C Locrian"
    \first_phrase_locrian

    \bar "|."
  }
  \layout { }
  \midi { }
}
