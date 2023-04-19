std_msgs/Header header

string note

# confidence in note classification *not confidence in overall detection*
float32 confidence

# loudness of relevant frequency(/ies), dB scaled
float32 loudness

# duration of note (optional, as determining it takes might take longer)
duration duration
