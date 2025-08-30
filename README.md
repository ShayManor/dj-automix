# dj-automix
System that mixes music from a library and takes prompts to make live changes (i.e. more energy, chill, harder, switch songs, etc.)

1) Ingest folder of songs and create csv with {title, artist, path, bpm?, key?}
2) Build fuse index for fuzzy text commands (“play bad bunny titi”)
3) Two decks, crossfade mix 
4) On first mix, run Essentia.js for bpm, downbeats, and key
5) Parse text into intents with the following methods:
- mix_in(title, bars=8) // Resolve title from fuse
- pause()
- fade_now()
- start(title)
- change_energy(delta_energy)
- key_move(same_minor, relative_minor) // change key, not sure how to do this.
- vocals(bool, mix_in_at=next_phrase + 16_bar_offset)