\version "2.24.1"
\include "swing.ly"

melody = \relative c' { c4 d e f g2 g a4 a a a g1 a4 a a a g1 f4 f f f e2 e g4 g g g c,1 }
first_phrase =                         \relative c' { c4 d e f g2 g a4 a a a g1 }
first_phrase_dorian =  \transpose d c  \relative c' { d4 e f g a2 a b4 b b b a1 }
% phrygian, lydian, mixolydian
first_phrase_minor =   \transpose a c' \relative c' { a4 b c d e2 e f4 f f f e1 }
first_phrase_locrian = \transpose b c' \relative c' { b4 c d e f2 f g4 g g g f1 }

\score {

\applySwing 8 #'(3 2) {
\relative c' {
c8 c d d e e f f g4 g g g a8 a a a a a a a g2 g2
}

\first_phrase

\transpose c e \first_phrase

\first_phrase_minor

\first_phrase_dorian

\first_phrase_locrian

}

\layout { }
\midi { \tempo 4 = 160 }
}
